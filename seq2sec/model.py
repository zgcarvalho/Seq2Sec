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
        self.act_01 = nn.CELU()
        self.conv_01 = nn.Conv1d(chan_in, 8, 1, padding=0)
        self.act_02 = nn.CELU()
        self.conv_02 = nn.Conv1d(8, chan_out, 3, padding=1)
        # """@nni.variable(nni.uniform(0.0, 0.9), name=p_drop)"""
        # p_drop = 0.3
        # self.dropout = nn.Dropout2d(p=p_drop)

    def forward(self, x):
        residual = x
        # out = self.conv_01(x)
        out = self.act_01(x)
        out = self.conv_01(out)
        # out = self.dropout(out)    
        out = self.act_02(out)
        out = self.conv_02(out)
        out += residual
        return out


class InceptionBlockLayer(nn.Module):

    def __init__(self, chan_in=24, chan_out=24):
        super(InceptionBlockLayer, self).__init__()
        self.act = nn.ReLU()

        self.conv_01a = nn.Conv1d(chan_in, 8, 1, padding=0)
        self.conv_02a = nn.Conv1d(8, chan_out, 3, padding=1)

        self.conv_01b = nn.Conv1d(chan_in, 8, 1, padding=0)
        self.conv_02b = nn.Conv1d(8, chan_out, 5, padding=2)

        self.maxpool_01c = nn.MaxPool1d(3, padding=1, stride=1)
        self.conv_02c = nn.Conv1d(chan_in, chan_out, 1)
        # """@nni.variable(nni.uniform(0.0, 0.9), name=p_drop)"""
        # p_drop = 0.3
        # self.dropout = nn.Dropout2d(p=p_drop)

    def forward(self, x):
        residual = x
        # out = self.conv_01(x)
        out = self.act(x)
        out_a = self.conv_01a(out)   
        out_a = self.act(out_a)
        out_a = self.conv_02a(out_a)
        out_b = self.conv_01b(out)   
        out_b = self.act(out_b)
        out_b = self.conv_02b(out_b)
        out_c = self.maxpool_01c(out)
        out_c = self.conv_02c(out_c)
        out_c = self.act(out_c)
        out = residual + out_a + out_b + out_c
        return out


class ExitLayer(nn.Module):

    def __init__(self, chan_in=24, chan_out=3):
        super(ExitLayer, self).__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, 1)

    def forward(self, x):
        return self.conv(x)

class ExitLayerReg(nn.Module):

    def __init__(self, chan_in=24, chan_hidden=4):
        super(ExitLayerReg, self).__init__()
        # self.conv_01 = nn.Conv1d(chan_in, chan_hidden, 1)
        # self.act_01 = nn.PReLU()
        # self.conv_02 = nn.Conv1d(chan_hidden, 1, 1)
        # self.act_02 = nn.ReLU()
        self.conv = nn.Conv1d(chan_in, 1, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x = self.conv_01(x)
        # x = self.act_01(x)
        # x = self.conv_02(x)
        # x = self.act_02(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class ResNet2(nn.Module):
    """output: example {'ss_cons_3_label':3}"""

    def __init__(self, tasks, n_blocks=21, chan_hidden=24):
        super(ResNet2, self).__init__()
        self.feat = FeatureLayer(chan_in=22, chan_out=chan_hidden)
        # self.block_layers = self._make_block_layers(n_blocks, chan_in=
        #     chan_hidden, chan_out=chan_hidden)
        self.block_layers = nn.ModuleList([InceptionBlockLayer(chan_in=chan_hidden, chan_out=chan_hidden) for i in range(n_blocks)])
        # self.exit_layers = {k: ExitLayer(chan_in=chan_hidden, chan_out=
            # output[k]) for k in output}
        self.exit_layers = self._make_exit_layers(tasks, chan_in=chan_hidden)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.
                    calculate_gain('leaky_relu'))

    @staticmethod
    def _make_exit_layers(tasks, chan_in):
        exit_layers = {}
        for t in tasks:
            if t == 'ss_cons_3_label':
                exit_layers[t] = ExitLayer(chan_in=chan_in, chan_out=3)
            elif t == 'ss_cons_4_label':
                exit_layers[t] = ExitLayer(chan_in=chan_in, chan_out=4)
            elif t == 'ss_cons_8_label':
                exit_layers[t] = ExitLayer(chan_in=chan_in, chan_out=8)
            elif t == 'buriedI_abs':
                exit_layers[t] = ExitLayerReg(chan_in=chan_in, chan_hidden=1)
        return exit_layers

    # @staticmethod
    # def _make_block_layers(n_blocks, chan_in, chan_out):
    #     layers = [BlockLayer(chan_in=chan_in, chan_out=chan_out) for i in
    #         range(n_blocks)]
    #     return nn.Sequential(*layers)

    def forward(self, x):
        """returns a dictionary where keys are task names and values are prediction values"""
        if self.training:
            x = self.feat(x)
            # x = self.block_layers(x)
            # switchs = np.random.randint(2,size=len(self.block_layers))
            switchs = np.random.binomial(1,np.linspace(1.0,0.5,len(self.block_layers))) 
            # count = 0
            for i in range(len(self.block_layers)):
                if switchs[i]:
                    x = self.block_layers[i](x)
                    # count += 1
            # print("# block layers ON = ", count)
            output = {k: self.exit_layers[k](x) for k in self.exit_layers}
            return output
        else:
            x = self.feat(x)
            # x = self.block_layers(x)
            for block in self.block_layers:
                x = block(x)
            output = {k: self.exit_layers[k](x) for k in self.exit_layers}
            return output

    def predict(self, x):
        """returns a list of lists. The inner list contains the prediction at each resnet block. The
        outer list separates the results per tasks"""
        b_results = [self.feat(x)]
        # list of results before exit layer. [0] is the result after feat layer. [1] is the result of 
        # the first block layer applied to [0]. [2] is the result of the second block layer to [1]...
        for block in self.block_layers:
            b_results.append(block(b_results[-1]))

        results = {}
        for k in self.exit_layers:
            results[k] = []
            if k == 'buriedI_abs':
                results[k] = [self.exit_layers[k](step).detach().numpy() for step in b_results]
            else:
                results[k] = [nn.functional.softmax(self.exit_layers[k](step), dim=1).detach().numpy() for step in b_results]
        return results

    def predict2(self, x):
        """returns a list of lists. The inner list contains the prediction at each resnet block. The
        outer list separates the results per tasks"""
        b_results = [self.feat(x)]
        # list of results before exit layer. [0] is the result after feat layer. [1] is the result of 
        # the first block layer applied to [0]. [2] is the result of the second block layer to [1]...
        # for block in self.block_layers:
        #     b_results.append(block(b_results[-1]))
        # switchs = np.random.randint(2,size=len(self.block_layers))
        switchs = np.random.binomial(1,np.linspace(1.0,0.5,len(self.block_layers))) 
            # count = 0
        for i in range(len(self.block_layers)):
            if switchs[i]:
                # x = self.block_layers[i](x)
                b_results.append(self.block_layers[i](b_results[-1]))
            else:
                b_results.append(b_results[-1])

        results = {}
        for k in self.exit_layers:
            results[k] = []
            if k == 'buriedI_abs':
                results[k] = [self.exit_layers[k](step).detach().numpy() for step in b_results]
            else:
                results[k] = [nn.functional.softmax(self.exit_layers[k](step), dim=1).detach().numpy() for step in b_results]
        return results

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.exit_layers = {k: self.exit_layers[k].to(*args, **kwargs) for k in self.exit_layers} 
        return self


def load(filename, device=torch.device('cpu')):
    net = torch.load(filename)
    net = net.to(device)
    net.eval()
    return net


if __name__ == '__main__':
    x = torch.randn((1, 22, 10))
    # print(x)
    # print(x.shape)
    # m = ResNet(21)
    # pred = m.predict(x)
    # print(pred)
    # print(len(pred))
    # mt = ResNetMT(21, chan_out=(3, 4))
    # y = mt(x)
    # print(y)
    # predmt = mt.predict(x)
    # print(predmt)
