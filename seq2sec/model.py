from torch import nn
import torch
import numpy as np

class FeatureLayer(nn.Module):
    def __init__(self, chan_in=22, chan_out=24):
        super(FeatureLayer, self).__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, 1, bias=False)

    def forward(self, x):
        return self.conv(x)

class BlockLayer(nn.Module):
    def __init__(self, chan_in=24, chan_out=24):
        super(BlockLayer, self).__init__()
        self.act_01 = nn.PReLU(chan_in)
        self.conv_01 = nn.Conv1d(chan_in, chan_out, 3, padding=1)
        self.act_02 = nn.PReLU(chan_in)
        # option to include a dropout here
        self.conv_02 = nn.Conv1d(chan_in, chan_out, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.act_01(x)
        out = self.conv_01(out)
        out = self.act_02(out)
        out = self.conv_02(out)
        out += residual
        return out

class ExitLayer(nn.Module):
    def __init__(self, chan_in=24, chan_out=3):
        super(ExitLayer,self).__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, 1)

    def forward(self, x):
        return self.conv(x)

# class ResNet(nn.Module):
#     def __init__(self, n_blocks, chan_out=3, chan_hidden=24):
#         super(ResNet,self).__init__()
#         self.feat = FeatureLayer(chan_in=22, chan_out=chan_hidden)
#         self.block_layers = self._make_block_layers(n_blocks, chan_in=chan_hidden, chan_out=chan_hidden)
#         self.exit = ExitLayer(chan_in=chan_hidden, chan_out=chan_out)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
#                 # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')

    
#     def _make_block_layers(self, n_blocks, chan_in, chan_out):
#         layers = [BlockLayer(chan_in=chan_in, chan_out=chan_out) for i in range(n_blocks)]
#         return nn.Sequential(*layers)
        

#     def forward(self, x):
#         x = self.feat(x)
#         x = self.block_layers(x)
#         x = self.exit(x)
#         return x

#     def predict(self, x):
#         x = self.feat(x)
#         results = [nn.functional.softmax(self.exit(block(x)), dim=1).detach().numpy() for block in self.block_layers]
#         return np.stack(results)

# class ResNetMT(nn.Module):
#     '''chan_out: tuple with channel number per exit layer. Ex (3,4, 8) means three exit layers
#     of a network capable to predict ss3, ss4 and ss8'''
#     def __init__(self, n_blocks, chan_out=(3,), chan_hidden=24):
#         assert(isinstance(chan_out, tuple))
#         super(ResNetMT,self).__init__()
#         self.feat = FeatureLayer(chan_in=22, chan_out=chan_hidden)
#         self.block_layers = self._make_block_layers(n_blocks, chan_in=chan_hidden, chan_out=chan_hidden)
#         # self.exit = ExitLayer(chan_in=chan_hidden, chan_out=chan_out)
#         self.exit_layers = [ ExitLayer(chan_in=chan_hidden, chan_out=x) for x in chan_out ]

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
#                 # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')

    
#     def _make_block_layers(self, n_blocks, chan_in, chan_out):
#         layers = [BlockLayer(chan_in=chan_in, chan_out=chan_out) for i in range(n_blocks)]
#         return nn.Sequential(*layers)
        

#     def forward(self, x):
#         '''returns a list of tensors where each tensor represents a task'''
#         x = self.feat(x)
#         x = self.block_layers(x)
#         x = [ exitlayer(x) for exitlayer in self.exit_layers ]
#         return x

#     def predict(self, x):
#         '''returns a list of lists. The inner list contains the prediction at each resnet block. The
#         outer list separates the results per tasks'''
#         x = self.feat(x)
#         #TODO this comprehension can be optimized. Same blocks are executed n=len(exit_layers) times.
#         #TODO include a softmax and change to np.array as done in ResNet.predict
#         results = [[ exitlayer(block(x)) for block in self.block_layers ] for exitlayer in self.exit_layers ]
#         return results

class ResNet2(nn.Module):
    '''output: example {'ss_cons_3_label':3}'''
    def __init__(self, output, n_blocks=21, chan_hidden=24):
        # assert(isinstance(chan_out, tuple))
        super(ResNet2,self).__init__()
        self.feat = FeatureLayer(chan_in=22, chan_out=chan_hidden)
        self.block_layers = self._make_block_layers(n_blocks, chan_in=chan_hidden, chan_out=chan_hidden)
        # self.exit = ExitLayer(chan_in=chan_hidden, chan_out=chan_out)
        # self.exit_layers = [ ExitLayer(chan_in=chan_hidden, chan_out=x) for x in chan_out ]
        self.exit_layers = {k: ExitLayer(chan_in=chan_hidden, chan_out=output[k]) for k in output}

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')

    
    def _make_block_layers(self, n_blocks, chan_in, chan_out):
        layers = [BlockLayer(chan_in=chan_in, chan_out=chan_out) for i in range(n_blocks)]
        return nn.Sequential(*layers)
        

    def forward(self, x):
        '''returns a dictionary where keys are task names and values are prediction values'''
        x = self.feat(x)
        x = self.block_layers(x)
        # x = [ exitlayer(x) for exitlayer in self.exit_layers ]
        output = {k: self.exit_layers[k](x) for k in self.exit_layers}
        return output

    # def predict(self, x):
    #     '''returns a list of lists. The inner list contains the prediction at each resnet block. The
    #     outer list separates the results per tasks'''
    #     x = self.feat(x)
    #     #TODO this comprehension can be optimized. Same blocks are executed n=len(exit_layers) times.
    #     #TODO include a softmax and change to np.array as done in ResNet.predict
    #     results = [[ exitlayer(block(x)) for block in self.block_layers ] for exitlayer in self.exit_layers ]
    #     return results

def load(filename):
    net = torch.load(filename) 
    net.eval()
    return net


if __name__ == "__main__":
    # CODE TO TEST
    x = torch.randn((1,22,10))
    print(x)
    print(x.shape)
    m = ResNet(21)
    # y = m(x)
    # print(y)
    # print(y.shape)
    pred = m.predict(x)
    print(pred)
    print(len(pred))
    # ResNetMT
    mt = ResNetMT(21, chan_out=(3,4))
    y = mt(x)
    print(y)
    predmt = mt.predict(x)
    print(predmt)
