import torch
from tensorboardX import SummaryWriter
from util import *

model = Net()
#model = torch.load('.log/model.pth')

simple_input = torch.Tensor([1,0])
batch_input = torch.Tensor([[1,0], [1,0]])

writer = SummaryWriter('runs/graph2', comment='Network')

with writer as w:
    w.add_graph(model, simple_input, True)

with writer as w:
    w.add_graph(model, batch_input, True)
