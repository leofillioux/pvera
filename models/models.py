import torch
import contextlib
import torch.nn as nn
from . import adapters
from timm.models import create_model

@contextlib.contextmanager
def local_seed(seed: int):
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_state)

class AbstractModel(nn.Module):
    def __init__(self, num_classes, config, adapter):
        super(AbstractModel, self).__init__()
        self.config = config
        self.adapter = adapter
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.prediction_head(x)
        return x
    
    def freeze(self):
        for name, param in self.backbone.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.prediction_head.parameters():
            param.requires_grad = True

class Model(AbstractModel):
    def __init__(self, base_model, num_classes, config, nb_grid_search, seed, adapter=None):
        super(Model, self).__init__(num_classes, config, adapter)
        self.config = config

        self.nb_grid_search = nb_grid_search
        self.seed = seed
        if base_model == 'DINOv2':
            print('Training DINOv2.')
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        else:
            raise NotImplementedError(f'Model {base_model} not implemented.')
        self.in_dim = self.backbone.norm.normalized_shape[0]
        self.prediction_head = adapters.GridSearchLinearProbe(self.nb_grid_search,
                                                              self.in_dim,
                                                              num_classes)
        self.add_adapter()
        self.freeze()
    
    def add_adapter(self):
        if self.adapter is None:
            self.grid_search = [0]
        
        elif self.adapter == 'lora':
            self.grid_search = self.config.lora.rank
            for encoder_block in self.backbone.blocks:
                with local_seed(self.seed):
                    encoder_block.attn.qkv = adapters.GridSearchLoRA(self.nb_grid_search,
                                                                     encoder_block.attn.qkv,
                                                                     encoder_block.attn.qkv.in_features,
                                                                     self.config.lora.rank,
                                                                     self.config.lora.alpha)
                    
        elif self.adapter == 'vera':
            self.grid_search = self.config.vera.lr
            rank = self.config.vera.rank

            q_downsample = [torch.empty(self.in_dim, rank) for _ in self.config.vera.lr]
            v_downsample = [torch.empty(self.in_dim, rank) for _ in self.config.vera.lr]
            q_upsample = [torch.empty(rank, self.in_dim) for _ in self.config.vera.lr]
            v_upsample = [torch.empty(rank, self.in_dim) for _ in self.config.vera.lr]

            with local_seed(self.seed):
                for qd, vd, qu, vu in zip(q_downsample, v_downsample, q_upsample, v_upsample):
                    nn.init.kaiming_uniform_(qd)
                    nn.init.kaiming_uniform_(vd)
                    nn.init.kaiming_uniform_(qu)
                    nn.init.kaiming_uniform_(vu)

            for encoder_block in self.backbone.blocks:
                with local_seed(self.seed):
                    encoder_block.attn.qkv = adapters.GridSearchVeRA(self.nb_grid_search,
                                                                     encoder_block.attn.qkv,
                                                                     self.in_dim,
                                                                     rank,
                                                                     self.config.vera.alpha,
                                                                     q_downsample,
                                                                     v_downsample,
                                                                     q_upsample,
                                                                     v_upsample)
                
        elif self.adapter == 'pvera':
            self.grid_search = self.config.pvera.lr
            rank = self.config.pvera.rank
            q_downsample = [torch.empty(self.in_dim, rank*2) for _ in self.config.pvera.lr]
            v_downsample = [torch.empty(self.in_dim, rank*2) for _ in self.config.pvera.lr]
            q_upsample = [torch.empty(rank, self.in_dim) for _ in self.config.pvera.lr]
            v_upsample = [torch.empty(rank, self.in_dim) for _ in self.config.pvera.lr]

            with local_seed(self.seed):
                for qd, vd, qu, vu in zip(q_downsample, v_downsample, q_upsample, v_upsample):
                    nn.init.kaiming_uniform_(qd)
                    nn.init.kaiming_uniform_(vd)
                    nn.init.kaiming_uniform_(qu)
                    nn.init.kaiming_uniform_(vu)

            for encoder_block in self.backbone.blocks:
                with local_seed(self.seed):
                    encoder_block.attn.qkv = adapters.GridSearchPVeRA(self.nb_grid_search,
                                                                      encoder_block.attn.qkv,
                                                                      self.in_dim,
                                                                      rank,
                                                                      self.config.pvera.alpha,
                                                                      q_downsample,
                                                                      v_downsample,
                                                                      q_upsample,
                                                                      v_upsample)
        elif self.adapter == 'adaptformer':
            self.grid_search = self.config.adaptformer.reduction_ratios
            for encoder_block in self.backbone.blocks:
                encoder_block.mlp = adapters.GridSearchAdaptFormer(self.nb_grid_search,
                                                                   encoder_block.norm2,
                                                                   encoder_block.mlp,
                                                                   self.config.adaptformer.activation,
                                                                   self.config.adaptformer.reduction_ratios)
                encoder_block.norm2 = nn.Identity()
        
        elif self.adapter == 'bottleneck':
            self.grid_search = self.config.bottleneck.reduction_ratios
            for encoder_block in self.backbone.blocks:
                encoder_block.attn = nn.Sequential(encoder_block.attn,
                                                   adapters.GridSearchBNAdapter(self.nb_grid_search,
                                                                                encoder_block.attn.proj.out_features,
                                                                                self.config.bottleneck.activation,
                                                                                self.config.bottleneck.reduction_ratios))
                encoder_block.mlp = nn.Sequential(encoder_block.mlp,
                                                  adapters.GridSearchBNAdapter(self.nb_grid_search,
                                                                               encoder_block.mlp.fc2.out_features,
                                                                               self.config.bottleneck.activation,
                                                                               self.config.bottleneck.reduction_ratios))

        elif self.adapter == 'dora':
            self.grid_search = self.config.dora.rank
            for encoder_block in self.backbone.blocks:
                encoder_block.attn.qkv = adapters.GridSearchDoRA(self.nb_grid_search,
                                                                 encoder_block.attn.qkv,
                                                                 encoder_block.attn.qkv.in_features,
                                                                 self.config.dora.rank,
                                                                 self.config.dora.alpha)

        elif self.adapter == 'ia3':
            self.grid_search = [0]
            for encoder_block in self.backbone.blocks:
                encoder_block.attn.qkv = adapters.IA3Attention(encoder_block.attn.qkv,
                                                               encoder_block.attn.qkv.in_features)
                encoder_block.mlp.act = adapters.IA3Linear(encoder_block.mlp.act,
                                                           encoder_block.mlp.fc1.out_features)
        else:
            raise NotImplementedError(f'Adapter {self.adapter} is not implemented.')
