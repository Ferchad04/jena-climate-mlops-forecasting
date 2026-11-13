import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class SegmentationTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img):
        image = TF.resize(
            img,
            self.size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        mask = TF.resize(
            img,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)
        mask = TF.pil_to_tensor(mask)

        mask = mask.squeeze(0) # [1, H, W] -> [H, W]
        mask = mask.long() # int64

        return image, mask


    def get_voc_loaders(
        data_dir,
        batch_size=8,
        image_size=(256, 256)
    ):
        transform = SegmentationTransform(
            image_size
        )

        train_dataset(
            root=data_dir,
            year='2012',
            image_set='train',
            download=True,
            transform=transform
        )

        val_dataset(
            root=data_dir,
            year='2012',
            image_set='val',
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader

# https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/
