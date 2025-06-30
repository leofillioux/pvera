import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

split_names_dict = {'train': 'train800.txt',
                    'val': 'val200.txt',
                    'test': 'test.txt'}

def parse_file_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' ', 1)
            result_dict[key] = int(value)
    return result_dict

class VTABDataset(Dataset):
    def __init__(self, name, config, split):
        super(VTABDataset, self).__init__()
        assert name in config.num_classes, f'Dataset {name} not supported.'
        assert split in ['train', 'val', 'test'], f'Split {split} not supported.'
        self.root_dir = os.path.join(config.data_dir, name)
        self.name_label_dict = parse_file_to_dict(os.path.join(self.root_dir, split_names_dict[split]))
        self.names = list(self.name_label_dict.keys())

        self.transforms = transforms.Compose([transforms.Resize(config.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.root_dir, img_name)).convert('RGB')
        img = self.transforms(img)
        label = self.name_label_dict[img_name]
        return img, label