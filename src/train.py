# -*- coding: utf-8 -*-
import torch
import data
import model

def train():
    # test data
    x = torch.randn(64,22,100)
    y = torch.randint(-1,3,(64,100), dtype=torch.long)

    # datasets
    trainset = data.SSDataset('config.json', use='training')
    valset = data.SSDataset('config.json', use='validation')

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=16, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16, shuffle=False)



    # model
    net = model.ResNet(21, chan_out=3, chan_hidden=24)

    # loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='elementwise_mean')

    # optimizer
    lr = 0.003
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # epochs
    for t in range(500):
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

if __name__ == "__main__":
    train()

