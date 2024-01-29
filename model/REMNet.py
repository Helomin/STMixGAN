import math
import random
import torch as t
from torch import nn
import torch.nn.functional as F


class convlstm_unit(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True):
        super(convlstm_unit, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels *
                              4, self.kernel_size, padding=self.padding, bias=self.bias)

    # input:
    # x [batch_size, in_channels, height, width]
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    # output:
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, x, state):
        h, c = state
        i, f, g, o = t.split(self.conv(t.cat((x, h), dim=1)),
                             self.hidden_channels, dim=1)
        i = t.sigmoid(i)
        f = t.sigmoid(f)
        o = t.sigmoid(o)
        g = t.tanh(g)
        c = f * c + i * g
        h = o * t.tanh(c)
        state = (h, c)
        return state


class convlstm_unit_remnet_cb(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, rec_length, bias=True):
        super(convlstm_unit_remnet_cb, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.rec_length = rec_length
        self.mem_gui_channels = int(240 / self.rec_length)
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels * 4, self.kernel_size,
                              padding=self.padding, bias=self.bias)
        self.conv_c = nn.Conv2d(self.hidden_channels, self.mem_gui_channels, (1, 1), (
            1, 1), bias=self.bias)  # convert the cell state c-1 to the query vector
        self.conv_fuse = nn.Conv2d(self.mem_gui_channels + self.hidden_channels, self.hidden_channels, (1, 1), (1, 1),
                                   bias=self.bias)  # 1x1 convolution kernel

    # c [batch_size, hidden_channels, height, width]
    # recalled_memory_features [batch_size, 240, 32, 32]
    # pertimestep_memory_guis [batch_size, mem_gui_channels, 32, 32]
    def pertimestep_perception_attention(self, c, recalled_memory_features):
        batch_size = c.size()[0]
        recalled_memory_features = recalled_memory_features.reshape(batch_size, self.rec_length, self.mem_gui_channels,
                                                                    16, 16)  # [batch_size, rec_length, mem_gui_channels, 32, 32]
        recalled_memory_features_averaged = F.adaptive_avg_pool3d(recalled_memory_features, (
            self.mem_gui_channels, 1, 1)).squeeze(4).squeeze(3)  # [batch_size, rec_length, mem_gui_channels]
        pertimestep_memory_guis = []
        # [batch_size, mem_gui_channels, height, width]
        query_vector = F.leaky_relu(self.conv_c(
            c), negative_slope=0.2, inplace=True)
        query_vector_averaged = F.adaptive_avg_pool2d(query_vector, (1, 1)).squeeze(
            3).squeeze(2)  # [batch_size, mem_gui_channels]
        for i in range(batch_size):
            attention_weight = t.cosine_similarity(query_vector_averaged[i].repeat(self.rec_length, 1),
                                                   recalled_memory_features_averaged[i], dim=1)  # [rec_length]
            attention_weight = t.unsqueeze(t.unsqueeze(t.unsqueeze(F.softmax(attention_weight, dim=0), dim=1), dim=2),
                                           dim=3)
            pertimestep_memory_gui = t.sum(t.mul(
                recalled_memory_features[i], attention_weight), dim=0)  # [mem_gui_channels, 32, 32]
            pertimestep_memory_guis.append(pertimestep_memory_gui)
        return t.stack(pertimestep_memory_guis, dim=0)

    # input:
    # x [batch_size, in_channels, height, width]
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    # recalled_memory_feature [batch_size, 240, 32, 32]
    # output:
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, x, state, recalled_memory_feature):
        h, c = state
        mt = self.pertimestep_perception_attention(c, recalled_memory_feature)
        i, f, g, o = t.split(self.conv(t.cat((x, h), dim=1)),
                             self.hidden_channels, dim=1)
        i = t.sigmoid(i)
        f = t.sigmoid(f)
        o = t.sigmoid(o)
        g = t.tanh(g)
        c = f * c + i * g
        h = o * F.tanh(self.conv_fuse(t.cat((c, mt), dim=1)))
        state = (h, c)
        return state


class convlstm_unit_remnet_fb(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, rec_length, bias=True):
        super(convlstm_unit_remnet_fb, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.rec_length = rec_length
        self.mem_gui_channels = int(240 / self.rec_length)
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels * 5, self.kernel_size,
                              padding=self.padding, bias=self.bias)
        self.conv_c = nn.Conv2d(self.hidden_channels, self.mem_gui_channels, (1, 1), (
            1, 1), bias=self.bias)  # convert the cell state c-1 to the query vector
        self.conv_ssm = nn.ConvTranspose2d(self.mem_gui_channels, self.mem_gui_channels, (4, 4), (2, 2), (1, 1),
                                           bias=self.bias)  # match spatial size 32 to 64
        self.conv_fuse = nn.Conv2d(self.mem_gui_channels + self.hidden_channels, self.hidden_channels, (1, 1), (1, 1),
                                   bias=self.bias)  # 1x1 convolution kernel

    # c [batch_size, hidden_channels, height, width]
    # recalled_memory_features [batch_size, 240, 32, 32]
    # pertimestep_memory_guis [batch_size, mem_gui_channels, 32, 32]
    def pertimestep_perception_attention(self, c, recalled_memory_features):
        batch_size = c.size()[0]
        recalled_memory_features = recalled_memory_features.reshape(batch_size, self.rec_length, self.mem_gui_channels,
                                                                    16, 16)  # [batch_size, rec_length, mem_gui_channels, 32, 32]
        recalled_memory_features_averaged = F.adaptive_avg_pool3d(recalled_memory_features, (
            self.mem_gui_channels, 1, 1)).squeeze(4).squeeze(3)  # [batch_size, rec_length, mem_gui_channels]
        pertimestep_memory_guis = []
        # [batch_size, mem_gui_channels, height, width]
        query_vector = F.leaky_relu(self.conv_c(
            c), negative_slope=0.2, inplace=True)
        query_vector_averaged = F.adaptive_avg_pool2d(query_vector, (1, 1)).squeeze(
            3).squeeze(2)  # [batch_size, mem_gui_channels]
        for i in range(batch_size):
            attention_weight = t.cosine_similarity(query_vector_averaged[i].repeat(self.rec_length, 1),
                                                   recalled_memory_features_averaged[i], dim=1)  # [rec_length]
            attention_weight = t.unsqueeze(t.unsqueeze(t.unsqueeze(F.softmax(attention_weight, dim=0), dim=1), dim=2),
                                           dim=3)
            pertimestep_memory_gui = t.sum(t.mul(
                recalled_memory_features[i], attention_weight), dim=0)  # [mem_gui_channels, 32, 32]
            pertimestep_memory_guis.append(pertimestep_memory_gui)
        return t.stack(pertimestep_memory_guis, dim=0)

    # input:
    # x [batch_size, in_channels, height, width]
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    # recalled_memory_feature [batch_size, 240, 32, 32]
    # output:
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, x, state, recalled_memory_feature):
        h, c = state
        mt = self.pertimestep_perception_attention(c, recalled_memory_feature)
        mt = F.leaky_relu(self.conv_ssm(mt), negative_slope=0.2, inplace=True)
        i, f, g, o, lam = t.split(
            self.conv(t.cat((x, h), dim=1)), self.hidden_channels, dim=1)
        i = t.sigmoid(i)
        f = t.sigmoid(f)
        o = t.sigmoid(o)
        g = t.tanh(g)
        lam = t.sigmoid(lam)
        c = f * c + i * g
        h_new = o * F.tanh(self.conv_fuse(t.cat((c, mt), dim=1)))
        h = lam * h_new + (1.0 - lam) * h
        state = (h, c)
        return state


class convlstm(nn.Module):
    # hidden_channels should be a list if the layers_num >= 2
    def __init__(self, in_channels, hidden_channels, kernel_size, layers_num, bias=True, return_all_layers=True, use_gpu=True):
        super(convlstm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = layers_num
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.use_gpu = use_gpu
        units_list = []
        for i in range(layers_num):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i-1]
            cur_hidden_channels = self.hidden_channels if layers_num == 1 else self.hidden_channels[
                i]
            units_list.append(convlstm_unit(
                cur_in_channels, cur_hidden_channels, self.kernel_size, bias=self.bias))
        self.units_list = nn.ModuleList(units_list)

    def zero_ini_layers_states(self, batch_size, height, width):
        ini_layers_states = []
        for i in range(self.layers_num):
            cur_hidden_channels = self.hidden_channels if self.layers_num == 1 else self.hidden_channels[
                i]
            zero_state = t.zeros(
                [batch_size, cur_hidden_channels, height, width])
            if self.use_gpu:
                zero_state = zero_state.cuda()
            zero_layer_states = (zero_state, zero_state)
            ini_layers_states.append(zero_layer_states)
        return ini_layers_states

    # input:
    # input [seq_len, batch_size, in_channels, height, width]
    # ini_layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # output:
    # layers_hidden_states [[seq_len, batch_size, hidden_channels, height, width],...]
    # layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, input, ini_layers_states=None):
        seq_len, batch_size, _, height, width = input.size()
        if ini_layers_states is None:
            ini_layers_states = self.zero_ini_layers_states(
                batch_size, height, width)
        layers_hidden_states = []
        layers_states = []
        cur_input = input
        for layer_index in range(self.layers_num):
            state = ini_layers_states[layer_index]
            layer_hidden_states = []
            for step in range(seq_len):
                state = self.units_list[layer_index](
                    cur_input[step, :, :, :, :], state)
                layer_hidden_states.append(state[0])
            layers_states.append(state)
            layer_hidden_states = t.stack(layer_hidden_states, dim=0)
            layers_hidden_states.append(layer_hidden_states)
            cur_input = layer_hidden_states
        if self.return_all_layers:
            return layers_hidden_states, layers_states
        else:
            return layers_hidden_states[-1], layers_states[-1]


class convlstm_remnet_cb(nn.Module):
    # hidden_channels should be a list if the layers_num >= 2
    def __init__(self, in_channels, hidden_channels, kernel_size, layers_num, rec_length, bias=True,
                 return_all_layers=True, use_gpu=True):
        super(convlstm_remnet_cb, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = layers_num
        self.rec_length = rec_length
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.use_gpu = use_gpu
        units_list = []
        for i in range(layers_num):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i-1]
            cur_hidden_channels = self.hidden_channels if layers_num == 1 else self.hidden_channels[
                i]
            units_list.append(
                convlstm_unit_remnet_cb(cur_in_channels, cur_hidden_channels, self.kernel_size, self.rec_length,
                                        bias=self.bias))
        self.units_list = nn.ModuleList(units_list)

    def zero_ini_layers_states(self, batch_size, height, width):
        ini_layers_states = []
        for i in range(self.layers_num):
            cur_hidden_channels = self.hidden_channels if self.layers_num == 1 else self.hidden_channels[
                i]
            zero_state = t.zeros(
                [batch_size, cur_hidden_channels, height, width])
            if self.use_gpu:
                zero_state = zero_state.cuda()
            zero_layer_states = (zero_state, zero_state)
            ini_layers_states.append(zero_layer_states)
        return ini_layers_states

    # input:
    # input [seq_len, batch_size, in_channels, height, width]
    # recalled_memory_feature [batch_size, 240, 16, 16]
    # ini_layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # output:
    # layers_hidden_states [[seq_len, batch_size, hidden_channels, height, width],...]
    # layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, input, recalled_memory_feature, ini_layers_states=None):
        seq_len, batch_size, _, height, width = input.size()
        if ini_layers_states is None:
            ini_layers_states = self.zero_ini_layers_states(
                batch_size, height, width)
        layers_hidden_states = []
        layers_states = []
        cur_input = input
        for layer_index in range(self.layers_num):
            state = ini_layers_states[layer_index]
            layer_hidden_states = []
            for step in range(seq_len):
                state = self.units_list[layer_index](
                    cur_input[step, :, :, :, :], state, recalled_memory_feature)
                layer_hidden_states.append(state[0])
            layers_states.append(state)
            layer_hidden_states = t.stack(layer_hidden_states, dim=0)
            layers_hidden_states.append(layer_hidden_states)
            cur_input = layer_hidden_states
        if self.return_all_layers:
            return layers_hidden_states, layers_states
        else:
            return layers_hidden_states[-1], layers_states[-1]


class convlstm_remnet_fb(nn.Module):
    # hidden_channels should be a list if the layers_num >= 2
    def __init__(self, in_channels, hidden_channels, kernel_size, layers_num, rec_length, bias=True,
                 return_all_layers=True, use_gpu=True):
        super(convlstm_remnet_fb, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = layers_num
        self.rec_length = rec_length
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.use_gpu = use_gpu
        units_list = []
        for i in range(layers_num):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i-1]
            cur_hidden_channels = self.hidden_channels if layers_num == 1 else self.hidden_channels[
                i]
            units_list.append(
                convlstm_unit_remnet_fb(cur_in_channels, cur_hidden_channels, self.kernel_size, self.rec_length,
                                        bias=self.bias))
        self.units_list = nn.ModuleList(units_list)

    def zero_ini_layers_states(self, batch_size, height, width):
        ini_layers_states = []
        for i in range(self.layers_num):
            cur_hidden_channels = self.hidden_channels if self.layers_num == 1 else self.hidden_channels[
                i]
            zero_state = t.zeros(
                [batch_size, cur_hidden_channels, height, width])
            if self.use_gpu:
                zero_state = zero_state.cuda()
            zero_layer_states = (zero_state, zero_state)
            ini_layers_states.append(zero_layer_states)
        return ini_layers_states

    # input:
    # input [seq_len, batch_size, in_channels, height, width]
    # recalled_memory_feature [batch_size, 240, 16, 16]
    # ini_layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # output:
    # layers_hidden_states [[seq_len, batch_size, hidden_channels, height, width],...]
    # layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, input, recalled_memory_feature, ini_layers_states=None):
        seq_len, batch_size, _, height, width = input.size()
        if ini_layers_states is None:
            ini_layers_states = self.zero_ini_layers_states(
                batch_size, height, width)
        layers_hidden_states = []
        layers_states = []
        cur_input = input
        for layer_index in range(self.layers_num):
            state = ini_layers_states[layer_index]
            layer_hidden_states = []
            for step in range(seq_len):
                state = self.units_list[layer_index](
                    cur_input[step, :, :, :, :], state, recalled_memory_feature)
                layer_hidden_states.append(state[0])
            layers_states.append(state)
            layer_hidden_states = t.stack(layer_hidden_states, dim=0)
            layers_hidden_states.append(layer_hidden_states)
            cur_input = layer_hidden_states
        if self.return_all_layers:
            return layers_hidden_states, layers_states
        else:
            return layers_hidden_states[-1], layers_states[-1]


class echo_lifecycle_encoder(nn.Module):
    def __init__(self):
        super(echo_lifecycle_encoder, self).__init__()
        self.conv3d_1 = nn.Conv3d(
            1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2), bias=True)
        self.conv3d_2 = nn.Conv3d(
            32, 64, (3, 3, 3), (1, 2, 2), (0, 1, 1), bias=True)
        self.conv3d_3 = nn.Conv3d(
            64, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_4 = nn.Conv3d(
            128, 256, (1, 3, 3), (1, 2, 2), (0, 1, 1), bias=True)
        self.linear = nn.Linear(256, 240, bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [batch_size, 240]
    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4)  # [batch_size, 1, in_seq_len, 256, 256]
        x = F.leaky_relu(self.conv3d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_4(x), negative_slope=0.2,
                         inplace=True)  # [batch_size, 256, 1, 16, 16]
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).squeeze(
            4).squeeze(3).squeeze(2)
        x = F.leaky_relu(self.linear(x), negative_slope=0.2, inplace=True)
        return x


class echo_motion_encoder(nn.Module):
    def __init__(self, in_seq_len):
        super(echo_motion_encoder, self).__init__()
        self.in_seq_len = in_seq_len
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=True)
        self.linear = nn.Linear(256 * (self.in_seq_len - 1), 240, bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [batch_size, 240]
    def forward(self, x):
        batch_size = x.size()[1]
        # [batch_size * (in_seq_len - 1), 1, 256, 256]
        diff_x = t.cat([x[i + 1] - x[i]
                       for i in range(self.in_seq_len - 1)], dim=0)
        diff_x = F.leaky_relu(self.conv2d_1(
            diff_x), negative_slope=0.2, inplace=True)
        diff_x = F.leaky_relu(self.conv2d_2(
            diff_x), negative_slope=0.2, inplace=True)
        diff_x = F.leaky_relu(self.conv2d_3(
            diff_x), negative_slope=0.2, inplace=True)
        # [batch_size * (in_seq_len - 1), 256, 16, 16]
        diff_x = F.leaky_relu(self.conv2d_4(
            diff_x), negative_slope=0.2, inplace=True)
        # [batch_size * (in_seq_len - 1), 256]
        diff_x = F.adaptive_avg_pool2d(diff_x, (1, 1)).squeeze()
        diff_x = diff_x.reshape(self.in_seq_len - 1, batch_size, -1)
        # [batch_size, 256 * (in_seq_len - 1)]
        diff_x = diff_x.permute(1, 0, 2).reshape(batch_size, -1)
        diff_x = F.leaky_relu(self.linear(
            diff_x), negative_slope=0.2, inplace=True)
        return diff_x


class query_vector_generator(nn.Module):
    def __init__(self):
        super(query_vector_generator, self).__init__()
        self.linear1 = nn.Linear(240, 240, bias=True)
        self.linear2 = nn.Linear(240, 240, bias=True)
        self.linear3 = nn.Linear(240, 240, bias=True)

    # input x1 [batch_size, 240] x2 [batch_size, 240]
    # output [batch_size, 240]
    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.linear1(x1), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.linear2(x2), negative_slope=0.2, inplace=True)
        return F.tanh(self.linear3(x1 + x2))


class perception_attention_mechanism(nn.Module):
    def __init__(self):
        super(perception_attention_mechanism, self).__init__()

    # input memory_pool [60, 240, 32, 32] query_vector [batch_size, 240]
    # output [batch_size, 240, 32, 32]
    def forward(self, memory_pool, query_vector):
        memory_pool_averaged = F.adaptive_avg_pool2d(
            memory_pool, (1, 1)).squeeze()  # [60, 240]
        recalled_memory_features = []
        batch_size = query_vector.size()[0]
        for i in range(batch_size):
            attention_weight = t.cosine_similarity(
                query_vector[i].repeat(60, 1), memory_pool_averaged, dim=1)
            attention_weight = t.unsqueeze(t.unsqueeze(t.unsqueeze(F.softmax(attention_weight, dim=0), dim=1), dim=2),
                                           dim=3)
            recalled_memory_feature = t.sum(
                t.mul(memory_pool, attention_weight), dim=0)
            recalled_memory_features.append(recalled_memory_feature)
        return t.stack(recalled_memory_features, dim=0)


class frame_encoder(nn.Module):
    def __init__(self):
        super(frame_encoder, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [in_seq_len, batch_size, 64, 64, 64]
    def forward(self, x):
        x = x.type(t.cuda.FloatTensor)
        in_seq_len, batch_size, channels, height, width = x.size()
        x = x.reshape(in_seq_len * batch_size, channels, height, width)
        x = F.leaky_relu(self.conv2d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_4(x), negative_slope=0.2, inplace=True)
        _, channels, height, width = x.size()
        x = x.reshape(in_seq_len, batch_size, channels, height, width)
        return x


class coarse_branch(nn.Module):
    def __init__(self, out_seq_len, use_gpu=True):
        super(coarse_branch, self).__init__()
        self.out_seq_len = out_seq_len
        self.use_gpu = use_gpu
        self.RNNs_encoder = convlstm(64, 128, 3, 1, use_gpu=self.use_gpu)
        self.RNNs_predictor = convlstm_remnet_cb(64, 128, 3, 1, self.out_seq_len, return_all_layers=False,
                                                 use_gpu=self.use_gpu)
        self.spatial_up_sampling = nn.ConvTranspose2d(
            128, 128, (4, 4), (2, 2), (1, 1), bias=True)

    def zero_RNNs_predictor_input(self, batch_size, channels, height, width):
        predictor_input = t.zeros(
            [self.out_seq_len, batch_size, channels, height, width])
        if self.use_gpu:
            predictor_input = predictor_input.cuda()
        return predictor_input

    def forward(self, x, recalled_memory_features):
        # Spatial Downsampling
        x = F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        # RNNs encoding
        _, encoded_states = self.RNNs_encoder(x)
        # RNNs predicting
        _, batch_size, channels, height, width = x.size()
        predictor_input = self.zero_RNNs_predictor_input(
            batch_size, channels, height, width)
        predicted_layers_hidden_states, _ = self.RNNs_predictor(predictor_input, recalled_memory_features,
                                                                encoded_states)
        # Spatial Upsampling
        seq_len, _, channels, height, width = predicted_layers_hidden_states.size()
        predicted_layers_hidden_states = predicted_layers_hidden_states.reshape(seq_len * batch_size, channels, height,
                                                                                width)
        predicted_layers_hidden_states = F.leaky_relu(self.spatial_up_sampling(predicted_layers_hidden_states),
                                                      negative_slope=0.2, inplace=True)
        _, channels, height, width = predicted_layers_hidden_states.size()
        predicted_layers_hidden_states = predicted_layers_hidden_states.reshape(seq_len, batch_size, channels, height,
                                                                                width)
        return predicted_layers_hidden_states


class fine_branch(nn.Module):
    def __init__(self, out_seq_len, use_gpu=True):
        super(fine_branch, self).__init__()
        # Temporal Downsampling
        self.out_seq_len = int(out_seq_len / 2)
        self.use_gpu = use_gpu
        self.RNNs_encoder = convlstm(64, 128, 3, 1, use_gpu=self.use_gpu)
        self.RNNs_predictor = convlstm_remnet_fb(64, 128, 3, 1, self.out_seq_len, return_all_layers=False,
                                                 use_gpu=self.use_gpu)

    def zero_RNNs_predictor_input(self, batch_size, channels, height, width):
        predictor_input = t.zeros(
            [self.out_seq_len, batch_size, channels, height, width])
        if self.use_gpu:
            predictor_input = predictor_input.cuda()
        return predictor_input

    def forward(self, x, recalled_memory_features):
        # RNNs encoding
        _, encoded_states = self.RNNs_encoder(x)
        # RNNs predicting
        _, batch_size, channels, height, width = x.size()
        predictor_input = self.zero_RNNs_predictor_input(
            batch_size, channels, height, width)
        predicted_layers_hidden_states, _ = self.RNNs_predictor(predictor_input, recalled_memory_features,
                                                                encoded_states)
        fb_output_feature_sequence = [
            predicted_layers_hidden_states[i].repeat(2, 1, 1, 1, 1)
            for i in range(self.out_seq_len)
        ]
        fb_output_feature_sequence = t.cat(fb_output_feature_sequence, dim=0)
        return fb_output_feature_sequence


class frame_decoder(nn.Module):
    def __init__(self):
        super(frame_decoder, self).__init__()
        self.deconv2d_1 = nn.ConvTranspose2d(
            256, 64, (4, 4), (2, 2), (1, 1), bias=True)
        self.deconv2d_2 = nn.ConvTranspose2d(
            64, 32, (4, 4), (2, 2), (1, 1), bias=True)
        self.deconv2d_3 = nn.ConvTranspose2d(
            32, 32, (3, 3), (1, 1), (1, 1), bias=True)
        self.deconv2d_4 = nn.ConvTranspose2d(32, 1, (1, 1), (1, 1), bias=True)

    # input [out_seq_len, batch_size, 256, 64, 64]
    # output [out_seq_len, batch_size, 1, 256, 256]
    def forward(self, x):
        out_seq_len, batch_size, channels, height, width = x.size()
        x = x.reshape(out_seq_len * batch_size, channels, height, width)
        x = F.leaky_relu(self.deconv2d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.deconv2d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.deconv2d_3(x), negative_slope=0.2, inplace=True)
        x = self.deconv2d_4(x)
        _, channels, height, width = x.size()
        x = x.reshape(out_seq_len, batch_size, channels, height, width)
        return x


class remnet(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, use_gpu=True):
        super(remnet, self).__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.use_gpu = use_gpu
        self.echo_lifecycle_encoder = echo_lifecycle_encoder()
        self.echo_motion_encoder = echo_motion_encoder(self.in_seq_len)
        self.query_vector_generator = query_vector_generator()
        self.lerm_memory_pool = nn.Parameter(t.randn(60, 240, 16, 16))
        self.perception_attention_mechanism = perception_attention_mechanism()
        self.frame_encoder = frame_encoder()
        self.coarse_branch = coarse_branch(self.out_seq_len, self.use_gpu)
        self.fine_branch = fine_branch(self.out_seq_len, self.use_gpu)
        self.frame_decoder = frame_decoder()

    def forward(self, x):
        x = x.type(t.cuda.FloatTensor)
        echo_motion_feature = self.echo_motion_encoder(x)
        echo_lifecycle_feature = self.echo_lifecycle_encoder(x)
        lerm_query_vector = self.query_vector_generator(
            echo_motion_feature, echo_lifecycle_feature)
        recalled_memory_features = self.perception_attention_mechanism(
            self.lerm_memory_pool, lerm_query_vector)  # [batchsize, 240, 32, 32]
        encoded_echo_frames = self.frame_encoder(x)
        cb_output_feature_sequence = self.coarse_branch(
            encoded_echo_frames, recalled_memory_features)
        fb_output_feature_sequence = self.fine_branch(
            encoded_echo_frames, recalled_memory_features)
        return self.frame_decoder(
            t.cat([cb_output_feature_sequence, fb_output_feature_sequence], dim=2)
        )


class remnet_sequence_discriminator(nn.Module):
    def __init__(self):
        super(remnet_sequence_discriminator, self).__init__()
        self.conv3d_1 = nn.Conv3d(
            1, 32, (3, 5, 5), (2, 2, 2), (1, 2, 2), bias=True)
        self.conv3d_2 = nn.Conv3d(
            32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_3 = nn.Conv3d(
            64, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_4 = nn.Conv3d(
            128, 256, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.linear = nn.Linear(256, 1)

    # input [out_seq_len, batch_size, 1, 256, 256]
    # output [batch_size]
    def forward(self, x):
        x = x.type(t.cuda.FloatTensor)
        x = x.permute(1, 2, 0, 3, 4)
        x = F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        x = F.leaky_relu(self.conv3d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_4(x), negative_slope=0.2, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).squeeze(
            4).squeeze(3).squeeze(2)
        x = F.sigmoid(self.linear(x)).squeeze(1)
        return x


class remnet_frame_patch_discriminator(nn.Module):
    def __init__(self):
        super(remnet_frame_patch_discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_5 = nn.Conv2d(256, 1, (3, 3), (2, 2), (1, 1), bias=True)

    # input [out_seq_len, batch_size, 1, 256, 256]
    # output [out_seq_len, batch_size, 8, 8]
    def forward(self, x):
        x = x.type(t.cuda.FloatTensor)
        out_seq_len, batch_size, channels, height, width = x.size()
        mean_x = t.stack([t.mean(t.stack([x[2 * i], x[2 * i + 1]]), dim=0)
                         for i in range(int(out_seq_len/2))], dim=0)
        mean_x = mean_x.reshape(int(out_seq_len/2) *
                                batch_size, channels, height, width)
        mean_x = F.leaky_relu(self.conv2d_1(
            mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_2(
            mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_3(
            mean_x), negative_slope=0.2, inplace=True)
        # [out_seq_len * batch_size, 256, 16, 16]
        mean_x = F.leaky_relu(self.conv2d_4(
            mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.tanh(self.conv2d_5(mean_x)).squeeze(
            1)  # [out_seq_len * batch_size, 8, 8]
        _, height, width = mean_x.size()
        # [out_seq_len, batch_size, 8, 8]
        mean_x = mean_x.reshape(int(out_seq_len/2), batch_size, height, width)
        return mean_x


def weighted_l1_loss(output, ground_truth, dataset='HKO_7'):
    if dataset == 'HKO_7':
        dBZ_ground_truth = 70.0 * ground_truth - 10.0
        weight_matrix = t.clamp(
            t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    elif dataset == 'Shanghai_2020':
        dBZ_ground_truth = 70.0 * ground_truth
        weight_matrix = t.clamp(
            t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    return t.mean(weight_matrix * t.abs(output - ground_truth))


def perceptual_similarity_loss(output, ground_truth, encoder, randomly_sampling=None):
    seq_len = output.size()[0]
    if randomly_sampling is not None:
        index = random.sample(range(seq_len), randomly_sampling)
        output_feature = encoder(output[index])
        ground_truth_feature = encoder(ground_truth[index])
    else:
        output_feature = encoder(output)
        ground_truth_feature = encoder(ground_truth)
    return t.mean(t.pow(t.abs(output_feature - ground_truth_feature), 2.0))


def fra_d_hinge_adv_loss(output, ground_truth, fra_d):
    real_fra = fra_d(ground_truth)
    fake_fra = fra_d(output)
    fra_d_loss_real = t.mean(t.relu(1.0 - real_fra))
    fra_d_loss_fake = t.mean(t.relu(1.0 + fake_fra))
    return fra_d_loss_real, fra_d_loss_fake


def seq_d_bce_adv_loss(input, output, ground_truth, seq_d):
    real_seq = seq_d(t.cat([input, ground_truth], dim=0))
    fake_seq = seq_d(t.cat([input, output], dim=0))
    seq_d_loss_real = F.binary_cross_entropy(real_seq, t.ones_like(real_seq))
    seq_d_loss_fake = F.binary_cross_entropy(fake_seq, t.zeros_like(fake_seq))
    return seq_d_loss_real, seq_d_loss_fake


def mixed_adversarial_loss(input, output, seq_d, fra_d):
    fake_seq = seq_d(t.cat([input, output], dim=0))
    fake_fra = fra_d(output)
    g_loss_seq = F.binary_cross_entropy(fake_seq, t.ones_like(fake_seq))
    g_loss_fra = -t.mean(fake_fra)
    return g_loss_seq + g_loss_fra
