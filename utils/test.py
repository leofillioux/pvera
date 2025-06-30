import os
import torch
import numpy as np
from tqdm import tqdm

def test(model, test_dataloader, metric, device, training, save_name):
    # load the best model
    model.load_state_dict(torch.load(os.path.join(training.save_dir, save_name, 'best_model.pth'), weights_only=True))
    model.eval()

    test_str = 'Test Loss {:.4f} | Top-1 accuracy {:.4f}'
    test_losses, test_probs, test_labs = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    for test_x, test_y in tqdm(test_dataloader, leave=False):
        with torch.no_grad():
            test_pred = model(test_x.to(device))
        test_loss = criterion(test_pred, test_y.to(device))
        test_losses.extend([test_loss.item()] * len(test_x))
        test_probs.extend(test_pred.cpu().numpy().tolist())
        test_labs.extend(test_y.cpu().numpy().tolist())
    test_loss = np.mean(test_losses)
    test_metric = metric(torch.tensor(np.array(test_probs)).argmax(1), torch.tensor(np.array(test_labs)))
    print(test_str.format(test_loss, test_metric))
    return test_metric, test_probs, test_labs