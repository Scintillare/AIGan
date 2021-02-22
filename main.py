import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AIGAN import AIGAN
from models import MNIST_target_net

if __name__ == "__main__":
    use_cuda=True
    image_nc=1
    epochs = 600
    batch_size = 128
    C_TRESH =  0.3
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./MNIST_target_model.pth"
    model = MNIST_target_net().to(device)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    model_num_labels = 10
    stop_epoch = 10


    # MNIST train dataset and dataloader declaration
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5) ]) # WARN don't do it
    transform = transforms.ToTensor()
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    # target = 4
    # labels = dataloader.dataset.targets
    # targets = torch.zeros_like(labels) + target 
    # dataloader.dataset.targets = targets

    aiGAN = AIGAN(device,
                    model,
                    model_num_labels,
                    image_nc,
                    stop_epoch,
                    BOX_MIN,
                    BOX_MAX,
                    C_TRESH,
                    dataset_name="mnist",
                    # is_targeted=True)
                    is_targeted=False)

    aiGAN.train(dataloader, epochs)