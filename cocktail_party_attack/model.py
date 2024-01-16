import torch.nn as nn
import json


class Network(nn.Module):
    def __init__(self, configration_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configration_file = configration_file
        with open(configration_file, 'r') as fp:
            self.config = json.load(fp)
        layer_arr = []
        for layer_dict in self.config:
            if layer_dict['layer'] == 'linear':
                layer_arr.append(nn.Linear(layer_dict['in_features'], layer_dict['out_features'],
                                           bias=layer_dict['bias']))
            elif layer_dict['layer'] == 'conv':
                layer_arr.append(nn.Conv2d(layer_dict['in_channels'], layer_dict['out_channels'],
                                           layer_dict['kernel_size'], stride=layer_dict['stride']
                                           , padding=layer_dict['padding'], bias=layer_dict['bias'],
                                           dilation=layer_dict['dilation'], groups=layer_dict['groups'],
                                           padding_mode=layer_dict['padding_mode']))
            elif layer_dict['layer'] == 'activation':
                if layer_dict['type'] == 'relu':
                    layer_arr.append(nn.ReLU())
        self.network = nn.Sequential(*layer_arr)

    def forward(self, inputs):
        return self.network(inputs)
