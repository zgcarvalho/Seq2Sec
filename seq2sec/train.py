# -*- coding: utf-8 -*-
import torch
from seq2sec import data
from seq2sec import model
from visdom import Visdom
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

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

    
    # training and validations losses, total and per tasks
    training_losses = {k:[] for k in tasks}
    training_losses['total'] = []
    validation_losses = {k:[] for k in tasks}
    validation_losses['total'] = []

    # metrics
    validation_acc = {k:[] for k in tasks}
    # validation_acc['total'] = []
    validation_balanced_acc = {k:[] for k in tasks}
    # validation_balanced_acc['total'] = []
    validation_cm = {k:[] for k in tasks}
    # validation_cm['total'] = []

    # epochs
    for e in range(10):
    
        # restart dictionary that accumulates losses per batch
        batch_of_losses = {k:[] for k in training_losses}

        # training loop
        net.train()
        for i, sample in enumerate(trainloader):
            # Forward pass: compute predicted y by passing x to the model.
            # print(sample['seq_res'])
            pred = net(sample['seq_res'])

            # Compute and print loss.
            losses = []
            for t in tasks:
                l = loss_fn(pred[t], sample[t])
                batch_of_losses[t].append(l.item())
                losses.append(l)

            # loss = loss_fn(y_pred, y)
            loss = sum(losses)
            # print(e, i, loss.item()) # print loss every batch (debug)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_of_losses['total'].append(loss.item())
        
        for k in training_losses:
            training_losses[k].append(np.mean(batch_of_losses[k]))
        
        
        # restart dictionary that accumulates losses per batch
        batch_of_losses = {k:[] for k in validation_losses}
        batch_of_acc = {k:[] for k in validation_losses}
        batch_of_balanced_acc = {k:[] for k in validation_losses}
        batch_of_cm = {k:[] for k in validation_losses}

        # validation loop
        net.eval()
        for i, sample in enumerate(valloader):
            # Forward pass: compute predicted y by passing x to the model.
            # print(sample['seq_res'])
            pred = net(sample['seq_res'])

            # Compute and print loss.
            losses = []
            for t in tasks:
                l = loss_fn(pred[t], sample[t])
                batch_of_losses[t].append(l.item())
                losses.append(loss_fn(pred[t], sample[t]))

                mask = sample[t].ge(0)
                y_true = torch.masked_select(sample[t], mask).numpy()
                y_pred = torch.masked_select(pred[t].argmax(dim=1), mask).numpy()

                acc = accuracy_score(y_true, y_pred)
                batch_of_acc[t].append(acc)

                bacc = balanced_accuracy_score(y_true, y_pred)
                batch_of_balanced_acc[t].append(bacc)
                # print("epoch: {} batch: {} task: {} acc: {} b_acc: {}".format(e, i, t, acc, bacc))
                cm = confusion_matrix(y_true,y_pred)
                batch_of_cm[t].append(cm)
                # print(cm)

            # loss = loss_fn(y_pred, y)
            loss = sum(losses)
            # print(e, i, loss.item()) # print loss every batch_of_losses = []

            batch_of_losses['total'].append(loss.item())
        
        for k in validation_losses:
            validation_losses[k].append(np.mean(batch_of_losses[k]))

        print("epoch: {} training_loss: {} validation_loss: {}".format(e, training_losses['total'][-1], validation_losses['total'][-1]))

        for t in tasks:
            validation_acc[t].append(np.mean(batch_of_acc[t]))
            validation_balanced_acc[t].append(np.mean(batch_of_balanced_acc[t]))
            print("-> task: {} acc: {} balanced_acc: {}".format(t, validation_acc[t][-1], validation_balanced_acc[t][-1]))


        
        


        

        # if e == 0:
        #     win_loss = viz.line(X=np.arange(0,len(training_losses)), Y=np.array(training_losses))
        # else: 
        #     viz.line(X=np.arange(0,len(training_losses)), Y=np.array(training_losses), win=win_loss, update='replace')

        # l = len(training_losses)
        # if e == 0:
        #     win_loss = viz.line(X=np.arange(0,l), Y=np.array(to_plot))
        # else:
        #     viz.line(X=np.arange(e*l,e*l+l), Y=np.array(to_plot), win=win_loss, update='append')
        # #     c+=1


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

