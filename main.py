import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN
from models import MNIST_target_net

if __name__ == "__main__":
    use_cuda=True
    image_nc=1
    epochs = 60
    batch_size = 128
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./MNIST_target_model.pth"
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()
    model_num_labels = 10
    stop_epoch = 10

    # MNIST train dataset and dataloader declaration
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    advGAN = AdvGAN(device,
                    target_model,
                    model_num_labels,
                    image_nc,
                    stop_epoch,
                    BOX_MIN,
                    BOX_MAX,
                    is_targeted=False)

    advGAN.train(dataloader, epochs)
