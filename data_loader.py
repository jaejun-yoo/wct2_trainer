import os
from torch.utils import data
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        root = os.path.join(root, "images", "trainval35k")
        self.image_paths = list(map(
            lambda x: os.path.join(root, x), os.listdir(root)
        ))
        self.transform = transform
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            data = self.transform(image)
        return data
    
    def __len__(self):
        return len(self.image_paths)

def get_loader(image_path, batch_size, num_workers=2, img_size=512):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dset = ImageFolder(image_path, transform=transform)
    loader = data.DataLoader(dataset=dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)

    return loader

def get_img(img_path, img_size=None):
    image = Image.open(img_path)
    lst = []
    if img_size is not None:
        lst.append(transforms.Resize((img_size, img_size)))
    lst.append(transforms.ToTensor())
    transform = transforms.Compose(lst)
    return transform(image).unsqueeze(0)