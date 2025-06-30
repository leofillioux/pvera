import os
import utils
import torch
import config
import models
import argparse
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from datasets import VTABDataset

parser = argparse.ArgumentParser(description='Train a model on a VTAB-1k dataset.')
parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset on which to train.', choices=config.dataset.num_classes.keys())
parser.add_argument('-m', '--model', type=str, required=True, help='Name of the model to train.', choices=['DINOv2'])
parser.add_argument('-a', '--adapter', type=str, default=None, help='Name of the adapter to use.', choices=config.adapters.keys())
args = parser.parse_args()

for seed in config.training.seeds:
    print(f'Seed: {seed}')
    utils.seed_everything(seed)

    # prepare the datasets
    train_dataloader = DataLoader(VTABDataset(args.dataset, config.dataset, 'train'),
                                    batch_size=config.training.batch_size,
                                    shuffle=False,
                                    num_workers=config.training.num_workers)

    val_dataloader = DataLoader(VTABDataset(args.dataset, config.dataset, 'val'),
                                batch_size=config.training.batch_size,
                                shuffle=False,
                                num_workers=config.training.num_workers)
    test_dataloader = DataLoader(VTABDataset(args.dataset, config.dataset, 'test'),
                                batch_size=config.training.batch_size,
                                shuffle=False,
                                num_workers=config.training.num_workers)

    # prepare the model
    nb_grid_search = utils.get_nb_grid_search(args.adapter, config.adapters)
    model = models.Model(args.model,
                         config.dataset.num_classes[args.dataset],
                         config.adapters,
                         nb_grid_search,
                         seed,
                         args.adapter)

    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Adapter: {args.adapter}')

    metric = Accuracy(top_k=1, task='multiclass', num_classes=config.dataset.num_classes[args.dataset])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')

    save_name = f'{args.dataset}_{args.model}_{args.adapter}_{seed}'
    results_path = os.path.join(config.training.results_dir, f'{save_name}.json')

    model, best_grid_search, best_epoch = utils.train(model, train_dataloader, val_dataloader, metric, device, config.training, args.adapter, save_name, nb_grid_search, args.dataset, config.adapters)
    test_metric, test_probs, test_labels = utils.test(model, test_dataloader, metric, device, config.training, save_name)
    utils.save_results(test_metric, test_probs, test_labels, best_grid_search, best_epoch, results_path)