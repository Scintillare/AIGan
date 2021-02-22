import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os

from torch.utils.tensorboard import SummaryWriter
import gc
from advertorch.attacks import LinfPGDAttack



# from confidence-calibrated-adversarial-training
def find_last_checkpoint(model_file):
    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = np.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adv_loss(probs_model, onehot_labels, is_targeted):   
    # C&W loss function
    real = torch.sum(onehot_labels * probs_model, dim=1)
    other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
    zeros = torch.zeros_like(other)
    if is_targeted:
        loss_adv = torch.sum(torch.max(other - real, zeros))
    else:
        loss_adv = torch.sum(torch.max(real - other, zeros))
    return loss_adv
    # or maximize cross_entropy loss
    # loss_adv = -F.mse_loss(logits_model, onehot_labels)
    # loss_adv = - F.cross_entropy(logits_model, labels)

class AIGAN:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 epoch_of_change,
                 box_min,
                 box_max,
                 c_tresh,
                 dataset_name,
                 is_targeted):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.c_treshold = c_tresh 
        self.dataset_name = dataset_name
        self.is_targeted = is_targeted
        
        self.models_path = './models/'
        self.writer = SummaryWriter('./checkpoints/logs/', max_queue=100)

        self.gen_input_nc = image_nc

        self.epoch_of_change = epoch_of_change
        self.attacker = LinfPGDAttack(self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=box_min, clip_max=box_max,
                targeted=self.is_targeted)

        if dataset_name=="mnist":
            from models import Generator, Discriminator
        elif dataset_name=="imagenet":
            from imagenet_models import PatchDiscriminator as Discriminator
            from imagenet_models import Resnet224Generator as Generator
        else:
            raise NotImplementedError('dataset [%s] is not implemented' % dataset_name)

        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)
        self.netG_file_name = self.models_path + 'netG.pth.tar'
        self.netDisc_file_name = self.models_path + 'netD.pth.tar'

        os.makedirs(self.models_path, exist_ok=True)

        # initialize all weights
        last_netG = find_last_checkpoint(self.netG_file_name)
        last_netDisc = find_last_checkpoint(self.netDisc_file_name)
        if last_netG is not None:
            self.netG.load_state_dict(torch.load(last_netG))
            self.netDisc.load_state_dict(torch.load(last_netDisc))
            *_, self.start_epoch = last_netG.split('.')
            self.iteration = None
            self.start_epoch = int(self.start_epoch)+1
        else:
            self.netG.apply(weights_init)
            self.netDisc.apply(weights_init)
            self.start_epoch = 1
            self.iteration = 0

       # initialize optimizers
        if self.dataset_name == "mnist":
            lr = 10**(-3)
        elif self.dataset_name == "imagenet":
            lr = 10**(-5)
        else:
            raise NotImplementedError('dataset [%s] is not implemented' % dataset_name)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=lr)
        self.optG_file_name = self.models_path + 'optG.pth.tar'
        self.optD_file_name = self.models_path + 'optD.pth.tar'

        last_optG = find_last_checkpoint(self.optG_file_name)
        last_optD = find_last_checkpoint(self.optD_file_name)
        if last_optG is not None:
            self.optimizer_G.load_state_dict(torch.load(last_optG))
            self.optimizer_D.load_state_dict((torch.load(last_optD)))

        self._use_attacker = (self.start_epoch < self.epoch_of_change)



    def train_batch(self, x, labels):
        # if training is targeted, labels = targets
       
        # optimize D
        for _ in range(1):
            # # add a clipping trick
            perturbation = torch.clamp(self.netG(x), -self.c_treshold, self.c_treshold)
            # perturbation = self.netG(x)
            adv_images = perturbation + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)
            
            self.optimizer_D.zero_grad()

            if self._use_attacker:
                pgd_images = self.attacker.perturb(x, labels) 
                d_real_logits, d_real_probs = self.netDisc(pgd_images)
            else:
                d_real_logits, d_real_probs = self.netDisc(x) 
            d_fake_logits, d_fake_probs = self.netDisc(adv_images.detach())

            # generate labels for discriminator (optionally smooth labels for stability)
            smooth = 0.0
            d_labels_real = torch.ones_like(d_real_probs, device=self.device) * (1 - smooth)
            d_labels_fake = torch.zeros_like(d_fake_probs, device=self.device)
            
            # discriminator loss
            loss_D_real = F.mse_loss(d_real_probs, d_labels_real)
            loss_D_real.backward()
            loss_D_fake = F.mse_loss(d_fake_probs, d_labels_fake)
            loss_D_fake.backward()
            loss_D_GAN = (loss_D_fake + loss_D_real) #/2
            # loss_D_GAN.backward()
            self.optimizer_D.step()
        
        gc.collect()

        # optimize G
        for _ in range(1):

            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            d_fake_logits, d_fake_probs = self.netDisc(adv_images.detach()) 
            loss_G_fake = F.mse_loss(d_fake_probs, torch.ones_like(d_fake_probs, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # # calculate perturbation norm
            loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            loss_perturb = torch.max(loss_perturb - self.c_treshold, torch.zeros(1, device=self.device))
            loss_perturb = torch.mean(loss_perturb)

            # cal adv loss
            # f_real_logits = self.model(x)
            # f_real_probs = F.softmax(f_real_logits, dim=1)
            f_fake_logits = self.model(adv_images) 
            f_fake_probs = F.softmax(f_fake_logits, dim=1)
            # if training is targeted, indicate how many examples classified as targets
            # else show accuraccy on adversarial images
            fake_accuracy = torch.mean((torch.argmax(f_fake_probs, 1) == labels).float())
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels.long()]
            loss_adv = adv_loss(f_fake_probs, onehot_labels, self.is_targeted)

            if self.dataset_name == "mnist":
                alambda = 1.
                alpha = 1.
                beta = 1.5
            elif self.dataset_name == "imagenet":
                alambda = 10.0#
                alpha = 1.
                beta = 0.5
            else:
                raise NotImplementedError('dataset [%s] is not implemented' % self.dataset_name)
            loss_G = alambda*loss_adv + alpha*loss_G_fake + beta*loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        self.writer.add_scalar('iter/train/loss_D_real', loss_D_real.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_D_fake', loss_D_fake.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_G_fake', loss_G_fake.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_perturb', loss_perturb.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_adv', loss_adv.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_G', loss_G.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/fake_acc', fake_accuracy.data, global_step=self.iteration)
        self.iteration += 1

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), loss_G.item(), fake_accuracy

    def train(self, train_dataloader, epochs):
        if self.iteration is None:
            self.iteration = (self.start_epoch-1)*len(train_dataloader)+1
        for epoch in range(self.start_epoch, epochs+1):
            if epoch == self.epoch_of_change:
                self._use_attacker = False
            if epoch == 120 and self.dataset_name == "mnist":
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 60 and self.dataset_name == "imagenet":
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=10**(-7))
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=10**(-7))
            if epoch == 200 and self.dataset_name == "mnist":
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            if epoch == 200  and self.dataset_name == "imagenet":
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=10**(-9))
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=10**(-9))
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_G_sum = 0
            fake_acc_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                gc.collect()
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                # # if targeted, create one hot vectors of the target
                # if self.is_targeted:
                #     assert(targets is not None)
                #     # this statement can be used when all targets is equal
                #     # targets = torch.zeros_like(labels) + target 
                #     # commmented because labels will be converted to one hot during training on batch  
                #     # labels = torch.eye(self.model_num_labels, device=self.device)[targets] #onehot targets 
                #     labels = targets

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_G_batch, fake_acc_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_G_sum += loss_G_batch
                fake_acc_sum += fake_acc_batch
                if i == len(train_dataloader)-2:
                    perturbation = self.netG(images)
                    self.writer.add_images('train/adversarial_perturbation', perturbation, global_step=epoch)
                    self.writer.add_images('train/adversarial_images', images+perturbation, global_step=epoch)
                    self.writer.add_images('train/adversarial_images_cl', torch.clamp(images+perturbation, self.box_min, self.box_max), global_step=epoch)


            # print statistics
            num_batch = len(train_dataloader)
            self.writer.add_scalar('epoch/train/loss_D', loss_D_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_G_fake', loss_G_fake_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_perturb', loss_perturb_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_adv', loss_adv_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_G', loss_G_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/fake_acc', fake_acc_sum/num_batch, global_step=epoch)

            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))
            
             # save generator
            if epoch%1==0:
                netG_file_name = self.netG_file_name + '.' + str(epoch) 
                torch.save(self.netG.state_dict(), netG_file_name)
                netD_file_name = self.netDisc_file_name + '.' + str(epoch) 
                torch.save(self.netDisc.state_dict(), netD_file_name)
                optG_file_name = self.optG_file_name + '.' + str(epoch) 
                torch.save(self.optimizer_G.state_dict(), optG_file_name)
                optD_file_name = self.optD_file_name + '.' + str(epoch) 
                torch.save(self.optimizer_D.state_dict(), optD_file_name)
            
        #save final model
        torch.save(self.netG.state_dict(), self.netG_file_name )
        torch.save(self.netDisc.state_dict(), self.netDisc_file_name)
        torch.save(self.optimizer_G.state_dict(), self.optG_file_name)
        torch.save(self.optimizer_D.state_dict(), self.optD_file_name)
