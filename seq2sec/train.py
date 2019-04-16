# -*- coding: utf-8 -*-
import torch
from seq2sec import data
from seq2sec import model
from visdom import Visdom
import numpy as np

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
# parser = argparse.ArgumentParser(description='Demo arguments')
# parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
#                     help='port the visdom server is running on.')
# parser.add_argument('-server', metavar='server', type=str,
#                     default=DEFAULT_HOSTNAME,
#                     help='Server address of the target to run the demo on.')
# FLAGS = parser.parse_args()

def train(data_config_file, fn_to_save_model=""):
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

    # # visdom example of text and update (to be removed later)
    # textwindow = viz.text('Hello World!')
    # updatetextwindow = viz.text('Hello World! More text should be here')
    # assert updatetextwindow is not None, 'Window was none'
    # viz.text('And here it is', win=updatetextwindow, append=True)

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

    training_losses = []
    validation_losses = []
    for e in range(10):
    
        # training loop
        batch_of_losses = []
        net.train()
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
            # print(e, i, loss.item()) # print loss every batch (debug)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_of_losses.append(loss.item())
        
        training_losses.append(np.mean(batch_of_losses))
        
        
        # validation loop
        batch_of_losses = []
        net.eval()
        for i, sample in enumerate(valloader):
            # Forward pass: compute predicted y by passing x to the model.
            # print(sample['seq_res'])
            pred = net(sample['seq_res'])

            # Compute and print loss.
            losses = []
            for t in tasks:
                losses.append(loss_fn(pred[t], sample[t]))

            # loss = loss_fn(y_pred, y)
            loss = sum(losses)
            # print(e, i, loss.item()) # print loss every batch_of_losses = []

            batch_of_losses.append(loss.item())
        
        validation_losses.append(np.mean(batch_of_losses))batch_of_losses = []

        
        l = len(training_losses)
        if e == 0:
            win_loss = viz.line(X=np.arange(0,l), Y=np.array(to_plot))
        else:
            viz.line(X=np.arange(e*l,e*l+l), Y=np.array(to_plot), win=win_loss, update='append')
        #     c+=1


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

