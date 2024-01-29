import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x


class ArgcLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ArgcLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)

        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)

        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)

        i = hard_sigmoid(wxhc[:, :self.out_channels])
        f = hard_sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = torch.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])

        c_1 = f * c_0 + i * g
        h_1 = i * torch.tanh(c_1)

        return h_1, (h_1, c_1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class Argc_PredNet(nn.Module):
    def __init__(self, use_gpu, R_channels, A_channels):
        super(Argc_PredNet, self).__init__()
        self.use_gpu = use_gpu
        self.r_channels = R_channels + [0, ]
        self.a_channels = A_channels
        self.n_layers = len(R_channels)

        for i in range(self.n_layers):
            cell = ArgcLSTMCell(2*self.a_channels[i]+self.r_channels[i+1],
                                self.r_channels[i],
                                (3, 3))
            setattr(self, f'cell{i}', cell)

        for l in range(self.n_layers):
            conv = nn.Sequential(
                ConvBlock(self.r_channels[l], self.r_channels[l])
            )
            if l == 0:
                conv.add_module('satlu', SatLU())

            setattr(self, f'conv{l}', conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(
                ConvBlock(2*self.a_channels[l], self.a_channels[l+1]),
                self.maxpool
            )
            setattr(self, f'update_A{l}', update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, f'cell{l}')
            cell.reset_parameters()

    def forward(self, input):

        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(
                batch_size, 2*self.a_channels[l], w, h))
            R_seq[l] = Variable(torch.zeros(
                batch_size, self.r_channels[l], w, h))

            if self.use_gpu:
                E_seq[l] = E_seq[l].cuda()
                R_seq[l] = R_seq[l].cuda()

            w = w//2
            h = h//2

        time_steps = input.size(1)
        for t in range(time_steps):
            A = input[:, t]
            if self.use_gpu:
                A = A.type(torch.cuda.FloatTensor)
            else:
                A = A.type(torch.FloatTensor)

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, f'cell{l}')

                E = E_seq[l]
                R = R_seq[l]

                hx = (R, R) if t == 0 else H_seq[l]

                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    R_up = self.upsample(R_seq[l+1])

                    tmp = torch.cat((E, R_up), dim=1)
                    R, hx = cell(tmp, hx)

                R_seq[l] = R
                H_seq[l] = hx

            for l in range(self.n_layers):
                conv = getattr(self, f'conv{l}')
                A_hat = conv(R_seq[l])

                if l == 0:
                    frame_prediction = A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)

                E = torch.cat([pos, neg], 1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, f'update_A{l}')
                    A = update_A(E)

        return frame_prediction


class SatLU(nn.Module):
    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return f'{self.__class__.__name__} (min_val={str(self.lower)}, max_val={str(self.upper)}{inplace_str})'


class DCCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*64, 1024),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv(x)
        x = self.linear(x)
        return x