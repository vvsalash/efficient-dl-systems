import os
import typing as tp
import zipfile
import gdown

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import Clothes, get_labels_dict


class ClothesDataset(Dataset):
    def __init__(self, folder_path, frame, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.frame = frame.set_index("image")
        self.img_list = list(self.frame.index.values)

        self.label2ix = get_labels_dict()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(f"{self.folder_path}/{img_name}.jpg").convert("RGB")
        img_transformed = self.transform(img)
        label = self.label2ix[self.frame.loc[img_name]["label"]]

        return img_transformed, label


def download_extract_dataset():
    if os.path.exists("clothing-dataset/images"):
        print("Dataset already exists")
        return

    raise RuntimeError("Dataset not found")


def get_train_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
