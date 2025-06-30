import os
import torch
import numpy as np
from tqdm import trange

def train(model, train_dataloader, val_dataloader, metric, device, config, adapter, save_name, nb_grid_search, dataset_name, adapter_config):
    if nb_grid_search == 1:
        optim_param_group = [{'params': model.prediction_head.heads.parameters(), **config.optimizer.params}]
        if adapter in ['lora', 'dora']:
            params = [list(model.backbone.blocks[i].attn.qkv.parameters()) for i in range(len(model.backbone.blocks))]
            params = [param for sublist in params for param in sublist]
            optim_param_group += [{'params': params, **config.optimizer.params}]
        elif adapter in ['vera', 'pvera']:
            params = [list(model.backbone.blocks[i].attn.qkv.parameters()) for i in range(len(model.backbone.blocks))]
            params = [param for sublist in params for param in sublist]
            optim_param_group += [{'params': params, 'lr': adapter_config[adapter].lr[0], 'weight_decay': 0.0001}]
        elif adapter in ['adaptformer']:
            params = [list(model.backbone.blocks[i].mlp.parameters()) for i in range(len(model.backbone.blocks))]
            params = [param for sublist in params for param in sublist]
            optim_param_group += [{'params': params, **config.optimizer.params}]
        elif adapter in ['ia3']:
            params = [list(model.backbone.blocks[i].attn.qkv.parameters()) for i in range(len(model.backbone.blocks))]
            params += [list(model.backbone.blocks[i].mlp.act.parameters()) for i in range(len(model.backbone.blocks))]
            params = [param for sublist in params for param in sublist]
            optim_param_group += [{'params': params, **config.optimizer.params}]
        elif adapter == 'bottleneck':
            params = [list(model.backbone.blocks[i].mlp[-1].parameters()) for i in range(len(model.backbone.blocks))]
            params += [list(model.backbone.blocks[i].attn[-1].parameters()) for i in range(len(model.backbone.blocks))]
            params = [param for sublist in params for param in sublist]
            optim_param_group += [{'params': params, **config.optimizer.params}]
    else:
        optim_param_group = [{'params': model.prediction_head.heads[i].parameters(), **config.optimizer.params} for i in range(nb_grid_search)]
        if adapter in ['lora', 'dora']:
            for j in range(nb_grid_search):
                params = [[p for p in model.backbone.blocks[i].attn.qkv.heads[j].parameters() if p.requires_grad]
                          for i in range(len(model.backbone.blocks))]
                params = [param for sublist in params for param in sublist]
                optim_param_group += [{'params': params, **config.optimizer.params}]
        elif adapter in ['vera', 'pvera']:
            for j in range(nb_grid_search):
                params = [[p for p in model.backbone.blocks[i].attn.qkv.heads[j].parameters() if p.requires_grad]
                          for i in range(len(model.backbone.blocks))]
                params = [param for sublist in params for param in sublist]
                optim_param_group += [{'params': params, 'lr': adapter_config[adapter].lr[j], 'weight_decay': 0.0001}]
        elif adapter in ['adaptformer']:
            for j in range(nb_grid_search):
                params = [[p for p in model.backbone.blocks[i].mlp.heads[j].parameters() if p.requires_grad]
                          for i in range(len(model.backbone.blocks))]
                params = [param for sublist in params for param in sublist]
                optim_param_group += [{'params': params, **config.optimizer.params}]
        elif adapter == 'bottleneck':
            for j in range(nb_grid_search):
                params = [[p for p in model.backbone.blocks[i].mlp[-1].heads[j].parameters() if p.requires_grad]
                          for i in range(len(model.backbone.blocks))]
                params += [[p for p in model.backbone.blocks[i].attn[-1].heads[j].parameters() if p.requires_grad]
                           for i in range(len(model.backbone.blocks))]
                params = [param for sublist in params for param in sublist]
                optim_param_group += [{'params': params, **config.optimizer.params}]

    optimizer = getattr(torch.optim, config.optimizer.name)(optim_param_group)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = 1e5
    best_grid_search_val_loss = [1e5 for _ in range(nb_grid_search)]
    best_grid_search_val_epoch = [-1 for _ in range(nb_grid_search)]
    best_grid_search_val_metrics = [0 for _ in range(nb_grid_search)]
    progress_str = 'Best epoch {} | Best val loss: {:.4f} | Best val metric: {:.4f} | Best grid search: {}'

    save_dir = os.path.join(config.save_dir, save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    progress = trange(config.num_epochs)
    for epoch in progress:
        # training
        model.train()
        for train_x, train_y in train_dataloader:
            train_x = train_x.repeat(nb_grid_search, 1, 1, 1)
            optimizer.zero_grad()
            train_pred_grid_search = model(train_x.to(device))
            if not isinstance(train_pred_grid_search, list):
                train_pred_grid_search = [train_pred_grid_search]
            total_loss = 0
            for i in range(nb_grid_search):
                if adapter in ['pvera']:
                    kld_loss = [model.backbone.blocks[j].attn.qkv.heads[i].kld for j in range(len(model.backbone.blocks))]
                    kld_loss = sum(kld_loss) / float(config.beta[dataset_name])
                else:
                    kld_loss = 0
                train_pred = train_pred_grid_search[i]
                train_loss = criterion(train_pred, train_y.to(device))
                total_loss += (train_loss+kld_loss)
            total_loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses, val_preds, val_labs = {i: [] for i in range(nb_grid_search)}, {i: [] for i in range(nb_grid_search)}, {i: [] for i in range(nb_grid_search)}
        for val_x, val_y in val_dataloader:
            with torch.no_grad():
                val_x = val_x.repeat(nb_grid_search, 1, 1, 1)
                val_pred_grid_search = model(val_x.to(device))
            if not isinstance(val_pred_grid_search, list):
                val_pred_grid_search = [val_pred_grid_search]
            for i in range(nb_grid_search):
                val_pred = val_pred_grid_search[i]
                val_loss = criterion(val_pred, val_y.to(device))
                val_losses[i].extend([val_loss.item()] * len(val_x))
                val_preds[i].extend(val_pred.argmax(dim=1).cpu().numpy())
                val_labs[i].extend(val_y.cpu().numpy())

        val_losses = [np.mean(val_losses[i]) for i in range(nb_grid_search)]
        val_metrics = [metric(torch.tensor(val_preds[i]), torch.tensor(val_labs[i])) for i in range(nb_grid_search)]
        for i in range(nb_grid_search):
            if (best_grid_search_val_loss[i] - val_losses[i])/(best_grid_search_val_loss[i]+1e-10)*100 > config.relative_epsilon:
                best_grid_search_val_loss[i] = val_losses[i]
                best_grid_search_val_epoch[i] = epoch
                best_grid_search_val_metrics[i] = val_metrics[i]

        if (best_val_loss - np.min(best_grid_search_val_loss))/(best_val_loss+1e-10)*100 > config.relative_epsilon:
            best_val_loss = np.min(best_grid_search_val_loss)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            epoch_best_grid_search = np.argmin(best_grid_search_val_loss)
            progress.set_description(progress_str.format(best_grid_search_val_epoch[epoch_best_grid_search] + 1,
                                                        best_grid_search_val_loss[epoch_best_grid_search],
                                                        best_grid_search_val_metrics[epoch_best_grid_search],
                                                        model.grid_search[epoch_best_grid_search]))
        
        # patience
        if all((epoch - np.array(best_grid_search_val_epoch)) >= config.patience):
            break
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=True))
    if nb_grid_search > 1:
        model.prediction_head.heads = model.prediction_head.heads[epoch_best_grid_search]
        if adapter in ['lora', 'vera', 'pvera', 'dora']:
            for i in range(len(model.backbone.blocks)):
                model.backbone.blocks[i].attn.qkv.heads = model.backbone.blocks[i].attn.qkv.heads[epoch_best_grid_search]
        elif adapter in ['bottleneck']:
            for i in range(len(model.backbone.blocks)):
                model.backbone.blocks[i].mlp[-1].heads = model.backbone.blocks[i].mlp[-1].heads[epoch_best_grid_search]
                model.backbone.blocks[i].attn[-1].heads = model.backbone.blocks[i].attn[-1].heads[epoch_best_grid_search]
        elif adapter in ['adaptformer']:
            for i in range(len(model.backbone.blocks)):
                 model.backbone.blocks[i].mlp.heads = model.backbone.blocks[i].mlp.heads[epoch_best_grid_search]
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    return model, model.grid_search[epoch_best_grid_search], best_grid_search_val_epoch[epoch_best_grid_search]