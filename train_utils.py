import torch
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler
from torchvision import transforms, datasets, models
import torch.nn as nn

# Data science tools
import numpy as np
# import pandas as pd

# Visualizations
import matplotlib.pyplot as plt

# Image manipulations
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import copy
from timeit import default_timer as timer

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_dataloaders(data_dir, valid_size, batch_size, image_transforms, show_sample=False):
    np.random.seed(12)
    torch.manual_seed(12)
    data = {
        'train':
        datasets.ImageFolder(data_dir, image_transforms['train']),
        'val':
        datasets.ImageFolder(data_dir, image_transforms['val']),
    }
    train_idx, valid_idx = [], []
    counts = (data['train'].targets.count(i) for i in data['train'].class_to_idx.values())
    acc = 0
    for numb in counts:
        valid_split = int(np.floor(valid_size * numb))
        indices = list(range(acc, acc+numb))
        np.random.shuffle(indices)
        acc += numb
        train_idx.extend(indices[:numb-valid_split])
        valid_idx.extend(indices[numb-valid_split:])

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    show_transform=False
    if show_transform:
        ex_path = os.path.join(data_dir , os.listdir(data_dir)[0])
        ex_path = os.path.join(ex_path, os.listdir(ex_path)[0])
        ex_img = Image.open(ex_path)
        imshow(ex_img)

        t = image_transforms['train']
        plt.figure(figsize=(24, 24))

        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            _ = imshow_tensor(t(ex_img), ax=ax)

        plt.tight_layout()
        plt.show()

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(data['train'],  batch_size=9, sampler=train_sampler,)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        plot_images(images, labels, data['train'].classes)

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'],  batch_size=batch_size, sampler=train_sampler, drop_last=True),
        'val': DataLoader(data['val'],  batch_size=batch_size, sampler=valid_sampler, drop_last=True),
    }
    return dataloaders



def train_model(model, 
        dataloaders, 
        criterion, 
        optimizer, 
        device, 
        save_file_name,
        max_epochs_stop=3,
        n_epochs=20,
        print_every=2,
        is_inception=False):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        dataloaders (PyTorch dataloader): dictionary with training and validation dataloaders 
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        device: torch.device - cuda or cpu
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
        is_inception: True if model is inception

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = {'acc': {'train': [], 'val': []},
                'loss': {'train': [], 'val': []}}

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):
        start = timer()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            phase_loss = 0.0
            phase_corrects = 0

            # Iterate over data.
            for ii, (data, labels) in enumerate(dataloaders[phase]):
                data = data.to(device)
                labels = labels.to(device)

                # zero the parameter gradients # Clear gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(data)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    # Track loss by multiplying average loss by number of examples in batch
                    phase_loss += loss.item() * data.size(0)
                    phase_corrects += torch.sum(preds == labels.data)
                    
                    # Track training progress
                    print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(dataloaders[phase]):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
            
            epoch_loss = phase_loss / len(dataloaders[phase].sampler)
            epoch_acc = phase_corrects.double() / len(dataloaders[phase].sampler)

            history['loss'][phase].append(epoch_loss)
            history['acc'][phase].append(epoch_acc)

            #  Print training and validation results
            if phase == 'val' and (epoch + 1) % print_every == 0:
                train_loss, valid_loss = history['loss']['train'][-1], history['loss']['val'][-1]
                train_acc, valid_acc = history['acc']['train'][-1], history['acc']['val'][-1]
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )
                
                plot_training_stats(history)
                # num_correct = 0
                # num_ex = 0
                # for data, labels in dataloaders['test']:
                #     data, labels = data.to(device), labels.to(device)
                #     probs = model(data)
                #     num_correct += sum((torch.argmax(probs, 1) == labels).float())
                #     num_ex += len(labels)
                # acc = num_correct/num_ex
                # print(acc)

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            
            # Save the model if validation loss decreases
            if phase == 'val' and epoch_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), r'./classifier/'+save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = epoch_loss
                valid_max_acc = epoch_acc
                best_epoch = epoch
            # Otherwise increment count of epochs with no improvement
            elif phase == 'val':
                epochs_no_improve += 1
                torch.save(model.state_dict(), r'./classifier/'+save_file_name+str(epoch))
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_max_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )


                    # Load the best state dict
                    model.load_state_dict(torch.load(r'./classifier/'+save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer
                    torch.save(model.state_dict(), r'./classifier/'+save_file_name)  # !!
                    return model, history
                             
        print()

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    torch.save(model.state_dict(), r'./classifier/'+save_file_name) 
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_max_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (n_epochs):.2f} seconds per epoch.'
    )
    
    return model, history


def plot_training_stats(history: dict):
    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
            history['loss'][c], label='loss '+c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Losses')
    # plt.show()
    plt.savefig("resnetloss.png")

    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
             history['acc'][c], label='acc ' + c)#100 *
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    # plt.show()
    plt.savefig("resnetacc.png")
            



# Auxilary ----------------------------------------------------


def plot_images(images, cls_true, label_names, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    X = images.numpy().transpose([0, 2, 3, 1])
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(np.clip(X[i, :, :, :], 0, 1), interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

