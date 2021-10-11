import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import dataset_nd as dataset
import evaluator
def Plot_loss(G_Loss, D_Loss, epoch = 20):
    import matplotlib.pyplot as plt
    plt.title('infogan Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(G_Loss)), G_Loss, label = 'generator')
    plt.plot(range(len(D_Loss)), D_Loss, label = 'discriminator')
    plt.legend()
    plt.savefig('./figure/' +"_loss_"+ str(epoch) + ".png")
    plt.close()
def Plot_acc(acc, epoch = 20):
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(range(len(acc)), acc, label = 'acc')
    plt.legend()
    plt.savefig('./figure/'+"acc_"+ str(epoch) + ".png")
    plt.close()

def Plot_new_acc(acc, epoch = 20):
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(range(len(acc)), acc, label = 'acc')
    plt.legend()
    plt.savefig('./figure/'+"new_acc_"+ str(epoch) + ".png")
    plt.close()
class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_size, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )  
        self.noise_size = noise_size
        
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(4, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
        )

        self.conditionToInput = nn.Linear(24, 64 * 64, bias=True)

    def forward(self, x, c):
        c = self.conditionToInput(c).view(-1, 1, 64, 64)
        #print(c.size())
        #print(x.size())
        x = torch.cat([x, c], 1)
        return self.main(x).view(-1, 1)

def sampleNoiseZ(batch_size, real_condition, noise_size, condition_size):
    z = torch.cat([torch.FloatTensor(batch_size, noise_size - condition_size).uniform_(-1.0, 1.0), real_condition.type(torch.float32).cpu()] , 1).view(-1, noise_size, 1, 1)
    return z.to(device), real_condition.type(torch.float32)


def runModel(batch_size, image_size, condition_size, noise_size, epochs, device, train_data, test_data, generator, discriminator, optim_G, optim_D, evaluation_model, criterion, new_test_data):
    total_acc = []
    new_total_acc = []
    loss_generator = []
    loss_discriminator = []
    iter_i = 0
    best_old_acc = 0
    best_new_acc = 0
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_LossG = 0
        total_LossD = 0
        for x, real_condition in train_data:
            iter_i += 1
            #print(iter_i)
            optim_D.zero_grad()
            optim_G.zero_grad()
            batch_size = x.size(0)
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #train with real
            real_condition = real_condition.to(device).type(torch.float32)
            real_x = x.to(device)
            classifier_label = torch.FloatTensor(batch_size,1).to(device)
            classifier_label.data.fill_(1)
            D_output_real = discriminator(real_x, real_condition)
            D_Loss_real = criterion(D_output_real, classifier_label)
            D_Loss_real.backward()
            D_x = D_output_real.mean().item()
            
            #train with fake
            #print(f"batch_size = {batch_size}")
            z, _ = sampleNoiseZ(batch_size, real_condition, noise_size, condition_size)
            fake_x = generator(z)
            D_output_fake = discriminator(fake_x.detach(), real_condition)
            classifier_label.data.fill_(0)
            D_Loss_fake = criterion(D_output_fake, classifier_label)
            D_Loss_fake.backward()
            D_G_z1 = D_output_fake.mean().item()
            D_Loss = D_Loss_real + D_Loss_fake
            optim_D.step()
            total_LossD += D_Loss.item()
            
            
            # (2) Update G network: maximize log(D(G(z)))
            D_output_fake_new = discriminator(fake_x, real_condition)
            classifier_label.data.fill_(1)
            D_G_z2 = D_output_fake_new.mean().item()
            G_Loss = criterion(D_output_fake_new, classifier_label)
            G_Loss.backward()
            optim_G.step()
            total_LossG += G_Loss.item()
            
            if (iter_i - 1) % 50 == 0:
                print(f"epoch / total_epochs --> {epoch}/{epochs}, iter/total_iter --> {iter_i-1}/{len(train_data)}:\n\tG_Loss: {G_Loss.item()}\n\tD_Loss:{D_Loss.item()}\n\tD_x:{D_x}\n\tD_G_z1:{D_G_z1}\n\tD_G_z2:{D_G_z2}")
                loss_generator.append(total_LossG)
                loss_discriminator.append(total_LossD)
        generator.eval()
        discriminator.eval()
        old_acc = 0
        new_acc_tmp = 0
        for real_condition in test_data:
            batch_size = len(real_condition)
            z = torch.cat([
            torch.FloatTensor(batch_size, noise_size - condition_size).uniform_(-1.0, 1.0), 
            torch.Tensor(real_condition.numpy())
            ] , 1).view(-1, noise_size, 1, 1).to(device)
            fake_x = generator(z)
            acc = evaluation_model.eval(fake_x, torch.Tensor(real_condition.numpy()))
            old_acc = acc
            detransformer = transforms.Compose([ 
                transforms.Normalize(mean = [ 0., 0., 0. ],
                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                     std = [ 1., 1., 1. ]),
               ]
            )        
            output_image = torch.randn(0, 3, 64, 64)
            for fake_image in fake_x:
                detransformed_image = detransformer(fake_image.cpu().detach())         
                output_image = torch.cat([output_image, detransformed_image.view(1, 3, 64, 64)], 0)

            save_image(make_grid(output_image, nrow=8), "tmp.jpg")
            total_acc.append(acc)
            print("acc: ", acc)
        
        for real_condition in new_test_data:
            batch_size = len(real_condition)
            z = torch.cat([
            torch.FloatTensor(batch_size, noise_size - condition_size).uniform_(-1.0, 1.0), 
            torch.Tensor(real_condition.numpy())
            ] , 1).view(-1, noise_size, 1, 1).to(device)
            fake_x = generator(z)
            new_acc = evaluation_model.eval(fake_x, torch.Tensor(real_condition.numpy()))
            new_acc_tmp = new_acc
            new_total_acc.append(new_acc)
            print("new acc: ", new_acc)
        if best_old_acc <= old_acc and best_new_acc < new_acc_tmp:
            best_old_acc = old_acc
            best_new_acc = new_acc_tmp
            torch.save(discriminator, "./model/" + 'best' + '_D')
            torch.save(generator, "./model/" + 'best' + '_G')
        if epoch % 3 == 0:
            Plot_loss(loss_generator, loss_discriminator, epoch)
            Plot_acc(total_acc, epoch)
            torch.save(discriminator, "./model/" + str(epoch) + '_D')
            torch.save(generator, "./model/" + str(epoch) + '_G')
            detransformer = transforms.Compose([ 
                transforms.Normalize(mean = [ 0., 0., 0. ],
                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                     std = [ 1., 1., 1. ]),
               ]
            )        
            output_image = torch.randn(0, 3, 64, 64)
            for fake_image in fake_x:
                n_image = detransformer(fake_image.cpu().detach())           
                output_image = torch.cat([output_image, n_image.view(1, 3, 64, 64)], 0)
            save_image(make_grid(output_image, nrow=8),"./figure/"+ 'generated_'+ str(epoch) + ".jpg")
            




if __name__ == '__main__':
    batch_size = 16
    image_size = 64
    LR_D = 2e-4
    LR_G = 1e-3
    
    condition_size = 24
    noise_size = 64
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DataLoader(dataset.ICLEVRLoader('./', mode = 'train'), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset.ICLEVRLoader('./', mode = 'test'), batch_size=32, shuffle=False)
    new_test_data = DataLoader(dataset.ICLEVRLoader('./', mode = 'new_test'), batch_size=32, shuffle=False)
    
    generator = Generator(noise_size).to(device)
    discriminator = Discriminator().to(device)
    
    optim_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    criterion_BCE = nn.BCELoss()
    evaluation_model = evaluator.evaluation_model()
    runModel(batch_size, image_size, condition_size, noise_size, epochs, device, train_data, test_data, generator, discriminator, optim_G, optim_D, evaluation_model, criterion_BCE, new_test_data)
    
    
    
    