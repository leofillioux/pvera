import os
import json

def save_results(test_metric, test_probs, test_labels, best_grid_search, best_epoch, save_path):
    results = {'test_metric': test_metric.item(),
               'best_grid_search': best_grid_search,
               'best_epoch': best_epoch,
               'test_probs': test_probs,
               'test_labels': test_labels}

    with open(save_path, 'w') as f:
        json.dump(results, f)