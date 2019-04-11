# -*- coding: utf-8 -*-
import torch
from seq2sec import data
from seq2sec import model

def train(data_config_file, fn_to_save_model=""):
    # test data
    # x = torch.randn(64,22,100)
    # y = torch.randint(-1,3,(64,100), dtype=torch.long)

    # datasets
    trainset = data.SSDataset(data_config_file, context='training')
    valset = data.SSDataset(data_config_file, context='validation')

    # get tasks to train
    tasks = trainset.tasks

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=16, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16, shuffle=False)

    # model
    net = create_model(tasks)

    # loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # optimizer
    lr = 0.003
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # epochs
    for e in range(10):
        for i, sample in enumerate(trainloader):
            # Forward pass: compute predicted y by passing x to the model.
            # print(sample['seq_res'])
            pred = net(sample['seq_res'])

            # Compute and print loss.
            losses = []
            for t in tasks:
                losses.append(loss_fn(pred[t], sample[t]))

            # loss = loss_fn(y_pred, y)
            loss = sum(losses)
            print(e, i, loss.item())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if fn_to_save_model != "":
        torch.save(net, fn_to_save_model)

def create_model(tasks):
    net_output = {}
    for t in tasks:
        if t == 'ss_cons_3_label':
            net_output[t] = 3
        elif t == 'ss_cons_4_label':
            net_output[t] = 4
        elif t == 'ss_cons_8_label':
            net_output[t] = 8
    
    return model.ResNet2(net_output)


if __name__ == "__main__":
    train('../data/config/data_test.json', '../models/teste_4.pth')

