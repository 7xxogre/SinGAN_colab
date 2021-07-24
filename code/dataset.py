import os
from glob import glob
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

class SinDataset(data.Dataset):

    def __init__(self, dir, transform):
        self.data_dir = dir
        self.transform = transform
        
        self.image_dir = sorted(glob(os.path.join(self.data_dir, '*.jpg')))[0]
    
    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        with open(self.image_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            return self.transform(img)


def get_dataset(args):
    # image processing
    train_transforms = transforms.Compose([transforms.Resize((256,256)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                std=[0.5, 0.5, 0.5])])
    
    val_transforms = transforms.Compose([transforms.Resize((256,256)), 
                                         transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                              std=[0.5, 0.5, 0.5])])

    train_dataset = SinDataset(args.data_dir, transform = train_transforms)
    val_dataset = SinDataset(args.data_dir, transform = val_transforms)

    return train_dataset, val_dataset
