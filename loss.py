import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn.functional as F

def log_cosh(x):
    return x + F.softplus(-2. * x) - math.log(2.0)

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()
    def forward(self, pred, target):
        return torch.sum(log_cosh(target - pred))

def gradient_penalty(D, real_imgs, fake_imgs, use_gpu):
    batch_size = real_imgs.size()[0]
    if real_imgs.ndim == 5:
        alpha = torch.rand(batch_size, 1, 1, 1, 1)
    else:
        alpha = torch.rand(batch_size, 1, 1, 1)
    if use_gpu:
        alpha = alpha.cuda()
    alpha = alpha.expand_as(real_imgs)
    interpolated = alpha * real_imgs.data + (1 - alpha) * fake_imgs.data
    interpolated = Variable(interpolated, requires_grad=True)
    if use_gpu:
        interpolated = interpolated.cuda()

    validity_interpolated = D(interpolated)

    grad_outputs = torch.ones(validity_interpolated.size())

    if use_gpu:
        grad_outputs = grad_outputs.cuda()

    gradients = grad(outputs=validity_interpolated, inputs=interpolated,
                       grad_outputs=grad_outputs,
                       create_graph=True, retain_graph=True)[0]

    gradients_norm = gradients.reshape(batch_size, -1).norm(p=2, dim=1)

    if use_gpu:
        gradients_norm = gradients_norm.cuda()

    return ((gradients_norm - 1) ** 2).mean()

def gradient_penalty_div(D, real_imgs, fake_imgs, p=6, k=2):
    real_imgs = Variable(real_imgs.type(torch.cuda.FloatTensor), requires_grad=True)
    fake_imgs = Variable(fake_imgs.type(torch.cuda.FloatTensor), requires_grad=True)

    real_grad_out = Variable(torch.Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False).type(torch.cuda.FloatTensor)
    real_validity = D(real_imgs)
    real_grad = torch.autograd.grad(
        real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p/2)

    fake_grad_out = Variable(torch.Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False).type(torch.cuda.FloatTensor)
    fake_validity = D(fake_imgs)
    fake_grad = torch.autograd.grad(
        fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p/2)

    return torch.mean(real_grad_norm + fake_grad_norm) * k / 2