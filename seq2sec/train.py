# -*- coding: utf-8 -*-
import torch
from seq2sec import data
from seq2sec import model

def train(netmodel, data_config_file, fn_to_save_model=""):
    # test data
    # x = torch.randn(64,22,100)
    # y = torch.randint(-1,3,(64,100), dtype=torch.long)

    # datasets
    trainset = data.SSDataset(data_config_file, use='training')
    valset = data.SSDataset(data_config_file, use='validation')

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=16, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16, shuffle=False)

    # model
    # net = model.ResNet(21, chan_out=4, chan_hidden=24)
    net = netmodel

    # loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='elementwise_mean')

    # optimizer
    lr = 0.003
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # epochs
    for t in range(10):
        for i, (x, y) in enumerate(trainloader):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = net(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            print(t, i, loss.item())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if fn_to_save_model != "":
        torch.save(net, fn_to_save_model)

if __name__ == "__main__":
    train('../data/config/data_test.json', '../models/teste_4.pth')

