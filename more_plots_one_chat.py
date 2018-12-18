import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from util import *

EPOCHS_TO_TRAIN = 5000  #50000

net = Net()
writer_train = SummaryWriter('runs/train_0')
writer_test = SummaryWriter('runs/test_0')

writer = SummaryWriter('runs/net_0')
writer.add_graph(net, torch.Tensor([[1,0]]), True)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update

        #########################################
        #ADD TRAINING LOSS TO SummaryWriter
        writer_train.add_scalar('LOSS', loss.data.item(), idx)

        for input, target in zip(inputs_test, targets_test):
            output = net(input)
            loss_test = criterion(output, target)

            #########################################
            #ADD TEST LOSS TO SummaryWriter
            writer_test.add_scalar('LOSS', loss_test.data.item(), idx)

        #check progress
        if idx%100==0:
            sys.stdout.write("Iterations: %d   \r" % (idx) )
            sys.stdout.flush()


print("Final results:")
test(inputs_test, targets_test, net)

print("Saving model")
net = torch.save(net, '.log/model.pth')
