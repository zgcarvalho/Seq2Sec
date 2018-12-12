from torch import nn
import torch

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

class ResNet(nn.Module):
    def __init__(self, n_blocks, chan_out=3, chan_hidden=24):
        super(ResNet,self).__init__()
        self.feat = FeatureLayer(chan_in=22, chan_out=chan_hidden)
        self.block_layers = self._make_block_layers(n_blocks, chan_in=chan_hidden, chan_out=chan_hidden)
        self.exit = ExitLayer(chan_in=chan_hidden, chan_out=chan_out)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')

    
    def _make_block_layers(self, n_blocks, chan_in, chan_out):
        layers = [BlockLayer(chan_in=chan_in, chan_out=chan_out) for i in range(n_blocks)]
        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.feat(x)
        x = self.block_layers(x)
        x = self.exit(x)
        return x

    def predict(self, x):
        x = self.feat(x)
        results = [self.exit(block(x)) for block in self.block_layers]
        return results


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
