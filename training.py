from seq2sec.train import train
from seq2sec.model import ResNet

if __name__ == "__main__":
    train(ResNet(21, chan_out=4, chan_hidden=24), 'data/config/data_test.json', 'models/teste_4.pth' )