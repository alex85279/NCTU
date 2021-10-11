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
import dataset
import argparse
import os
import time
import math
import argparse
import pprint


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import torch.nn.utils as utils
def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.
    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor
def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.
    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)
class AverageMeter(object):
    """Computes and stores the average and current value.
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
class ActNorm(nn.Module):
    """Activation normalization for 2D inputs.
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x
class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = (x_change - t) * s.mul(-1).exp()
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change) * s.exp() + t
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_condconv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_condconv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, x_cond):
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = F.relu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x
class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj

class Glow(nn.Module):

    def __init__(self, num_channels, num_levels, num_steps, mode='sketch'):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           cond_channels=4,
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)
        self.mode = mode
        #self.condToInputSize = nn.Linear(40,128*128)
        self.condToInputSize = nn.Sequential(
            nn.Linear(40,250,bias=False),
            nn.Linear(250,1000,bias=False),
            nn.Linear(1000,128*128,bias=False)
        )
        #self.condition_embedding = nn.Embedding(40, 64*64)
    def forward(self, x, x_cond, reverse=False):
        x_cond = self.condToInputSize(x_cond).view(-1,1,128,128)
        #x_cond = torch.LongTensor([x_cond]).to(device)
        #x_cond = self.condition_embedding(x_cond)
        #print(x_cond)
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)
        if self.mode == 'gray':
            x_cond, _ = self._pre_process(x_cond)
        
        x = squeeze(x)
        x_cond = squeeze(x_cond)
        #print(x.size())
        #print(x_cond.size())
        x, sldj = self.flows(x, x_cond, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        # y = x
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels=cond_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              cond_channels=4 * cond_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, x_cond, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x_cond, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x_cond = squeeze(x_cond)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x_cond, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x_cond = squeeze(x_cond, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x_cond, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, cond_channels, mid_channels)

    def forward(self, x, x_cond, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x_cond, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, x_cond, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.
    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.
    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x











def runModel(batch_size, image_size, condition_size, noise_size, epochs, device, train_data, model, optimizer, train_dataset):
    total_acc = []
    loss_all = []
    iter_i = 0
    loss_fn = NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    loss_meter = AverageMeter()
    for epoch in range(epochs):
        model.train()
        total_Loss = 0
        iter_i = 0
        for batch in train_data:
            x, real_condition = batch
            #print(x.size())
            #print(real_condition.size())
            x = x.to(device)
            #print(real_condition.size())
            iter_i += 1
            #print(iter_i)
            optimizer.zero_grad()
            batch_size = x.size(0)
            #print(x)
            real_condition = real_condition.to(device).type(torch.float32)
           
            #conditional glow
            z, sldj = model(x, real_condition,reverse=False)
            
            loss = loss_fn(z,sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            loss_all.append(loss)
            
            clip_grad_norm(optimizer, 50)
            #print(real_x)
            
            #loss = - model.log_prob(real_x, real_condition, bits_per_pixel=True).mean(0)
            #print(f"epoch:{epoch}. iter:{iter_i-1}\tloss: {loss.item()}")
            #loss = loss + L2_loss / 256
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            
            if (iter_i - 1) % 50 == 0:
                print(f"epoch / total_epochs --> {epoch}/{epochs}, iter/total_iter --> {iter_i-1}/{len(train_data)}:\n\tLoss: {loss.item()}")
                loss_all.append(loss)
                #test generate
                model.eval()
                tmp_condition = real_condition
                real_condition = real_condition.to(device).type(torch.float32)
                generate_image(model, real_condition, epoch)
                '''sigma = 1
                z = torch.randn((32, 3, 64, 64), dtype=torch.float32, device=device) * sigma
                fake_x, _ = model(z, real_condition, reverse=True)
                fake_x = torch.sigmoid(fake_x)
                transform_image = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                output_image = torch.randn(0,3,64,64)
                for fake_image in fake_x:
                    output_image = torch.cat([output_image, transform_image(fake_image.cpu().detach()).view(1,3,64,64)],0)
                output_image = output_image.to(device)'''
                ##############
                    
                    
                '''save_image(make_grid(fake_x, nrow=8),"./" + "tmp.jpg")
                save_image(make_grid(x, nrow=8),"./" + "tmp_x.jpg")'''
                    
                model.train()
        model.eval()
        interpolation_image(model, train_dataset, [0,2,4,6,8], [1,3,5,7,9], device, epoch)  
        attribute_interpolation(model, train_dataset, [0], device, 999, train_data, 39)
        attribute_interpolation(model, train_dataset, [0], device, 998, train_data, 36)        

        if epoch % 3 == 0:
            Plot_loss(loss_all, epoch)
            torch.save(model, "./model/" + str(epoch) + '_cG')

def generate_image(model, condition, epoch):
    sigma = 1
    z = torch.randn((32, 3, 128, 128), dtype=torch.float32, device=device) * sigma
    fake_x, _ = model(z, condition, reverse=True)
    fake_x = torch.sigmoid(fake_x)
    save_image(make_grid(fake_x, nrow=8),"./figure/" + "generated_" + str(epoch) + ".jpg")
def interpolation_image(model, train_dataset, idx1, idx2, device, epoch):
    output_image = torch.randn(0, 3, 128, 128)
    for i in range(5):
        i1 = idx1[i]
        i2 = idx2[i]
        image1, condition1 = train_dataset[i1]
        image2, condition2 = train_dataset[i2]
        image1 = image1.to(device).view(1,3,128,128)
        image2 = image2.to(device).view(1,3,128,128)
        condition1 = torch.from_numpy(condition1)
        condition2 = torch.from_numpy(condition2)
        condition1 = condition1.to(device).type(torch.float32).view(1,40)
        condition2 = condition2.to(device).type(torch.float32).view(1,40)
    
        z1,_ = model(image1, condition1, reverse=False)
        z2,_ = model(image2, condition2, reverse=False)
        z_inv = (z2 - z1) / 8
        c_inv = (condition2 - condition1) / 8
        for i in range(8):
            z = z1 + z_inv * i;
            c = condition1 + c_inv * i
            
            fake_image,_ = model(z,c,reverse=True)
            fake_image = torch.sigmoid(fake_image)
            output_image = torch.cat([output_image, fake_image.view(1,3,128,128).cpu()],0)
    save_image(make_grid(output_image, nrow=8),"./figure/"+ 'interpolate_'+ str(epoch) + ".jpg")
def attribute_interpolation(model, train_dataset, idx, device, epoch, train_data_loader, attribute_num):
    #39,36
    pos_sample_list = [1, 2, 3, 5, 8]
    neg_sample_list = [23, 49, 52, 62, 80]
    image_num = 5
    pos_sample = torch.zeros((1,3,128,128), dtype=torch.float32, device = device)
    neg_sample = torch.zeros((1,3,128,128), dtype=torch.float32, device = device)
    for i in range(image_num):
        pos_input, pos_condition = train_dataset[pos_sample_list[i]]
        neg_input, neg_condition = train_dataset[neg_sample_list[i]]
        pos_input = pos_input.to(device).view(1,3,128,128)
        neg_input = neg_input.to(device).view(1,3,128,128)
        pos_condition = torch.from_numpy(pos_condition).to(device).type(torch.float32).view(1,40)
        neg_condition = torch.from_numpy(neg_condition).to(device).type(torch.float32).view(1,40)
        
        pos_latent_z,_ = model(pos_input, pos_condition, reverse = False)
        neg_latent_z,_ = model(neg_input, neg_condition, reverse = False)
        pos_latent_z = pos_latent_z / image_num
        neg_latent_z = neg_latent_z / image_num
        pos_sample.add_(pos_latent_z)
        neg_sample.add_(neg_latent_z)
    
    i1 = idx[0]
    output_image = torch.randn(0, 3, 128, 128)
    image, condition = train_dataset[i1]
    image = image.to(device).view(1,3,128,128)
    condition = torch.from_numpy(condition)
    condition = condition.to(device).type(torch.float32).view(1,40)
    neg_condition = torch.clone(condition)
    neg_condition[0,attribute_num] = -1
    #neg_condition[0,39] = -1
    pos_condition = torch.clone(condition)
    pos_condition[0,attribute_num] = 1
    #pos_condition[0,39] = 1
        

    z_inv = (pos_sample - neg_sample) / 8
    c_inv = (pos_condition - neg_condition) / 8
    for i in range(8):
        z = neg_sample + z_inv * i
        c = neg_condition + c_inv * i
        fake_image,_= model(z,c,reverse=True)
        fake_image = torch.sigmoid(fake_image)
        output_image = torch.cat([output_image, fake_image.view(1,3,128,128).cpu()],0)
    save_image(make_grid(output_image, nrow=8),"./figure/"+ 'attributeInt_'+ str(epoch) + ".jpg")
def Plot_loss(Loss, epoch = 20):
    import matplotlib.pyplot as plt
    plt.title('C-glow Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(Loss)), Loss, label = 'c-glow-model')
    plt.legend()
    plt.savefig('./figure/' +"_loss_"+ str(epoch) + ".png")
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train c-Glow')

    


    args = parser.parse_args()
    batch_size = 32
    image_size = 128
    LR_D = 2e-4
    LR_G = 0.002
    
    condition_size = 24
    noise_size = 128
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = dataset.CelebALoader('./')
    
    
    #device = torch.device("cpu")
    train_data = DataLoader(dataset.CelebALoader('./'), batch_size=batch_size, shuffle=True)
    #test_data = DataLoader(dataset.ICLEVRLoader('./images/', mode = 'test'), batch_size=32, shuffle=False)
    #def __init__(self, width, depth, n_levels, input_dims=(4,64,64), checkpoint_grads=False, lu_factorize=False):
    #model = Glow(256, 4, 3).to(device)
    #model = CondGlowModel(args).to(device)
    model = Glow(32, 3, 8, "sketch").to(device)
    #model.load_state_dict(torch.load("./model/24_cG.pt"))
    
    #attribute_interpolation(model, train_dataset, [0], device, 999, train_data, 39)
    optimizer = optim.Adam(model.parameters(), lr=LR_G, betas=(0.5, 0.999))
    

    runModel(batch_size, image_size, condition_size, noise_size, epochs, device, train_data, model, optimizer, train_dataset)
    
    
    
    #interpolation_image(model, train_data, [0,2,4,6,8], [1,3,5,7,9], device)
    #attribute_interpolation(model, train_data, [0,1,2,3,4], device)