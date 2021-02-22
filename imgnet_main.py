import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from AIGAN import AIGAN
import torch.nn as nn
# Image manipulations
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from train_utils import *
from imagenet_models import get_resnet50 as Resnet50

if __name__ == "__main__":
    torch.cuda.empty_cache()
    use_cuda=True
    image_nc=3
    epochs = 600
    batch_size = 10
    C_TRESH =  0.3 #8
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    data_dir = r'D:\'
    stop_epoch = 1

    image_transforms = {
        #     # Train uses data augmentation
            'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=299),#, scale=(1., 1.0)
                transforms.RandomResizedCrop(size=224),#, scale=(1., 1.0)
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
                # transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #WARN don`t do it!!
            ]),
        #     # Validation does not use augmentation
            'val':
            transforms.Compose([
                # transforms.Resize(size=(350)),
                # transforms.CenterCrop(299),
                transforms.Resize(size=(256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #WARN don`t do it!!
            ]),
        }

    fm = "pretr_resnet.pt"
    model_num_labels = 2
    model = Resnet50(model_num_labels)
    model.load_state_dict(torch.load(fm))
    model.to(device)
    model.eval()


    trainset = datasets.ImageFolder(data_dir, transform=image_transforms['train'])  
    dataloader_train = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)

    aiGAN = AIGAN(device,
                            model,
                            model_num_labels,
                            image_nc,
                            stop_epoch,
                            BOX_MIN,
                            BOX_MAX,
                            C_TRESH,
                            dataset_name="imagenet",
                            is_targeted=False)

    aiGAN.train(dataloader_train, epochs)