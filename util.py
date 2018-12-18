import torch
from torch.autograd import Variable
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4, False)
        self.fc2 = nn.Linear(4, 1, False)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


#TRAIN DATA
inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))

#TEST DATA
inputs_test = list(map(lambda s: Variable(torch.Tensor([s])), [[1.0002, 0.0001]]))
targets_test = list(map(lambda s: Variable(torch.Tensor([s])), [[1]]))

#TEST
def test(inputs, targets, net):
    for input, target in zip(inputs, targets):
        output = net(input)
        print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
            int(input.data.numpy()[0][0]),
            int(input.data.numpy()[0][1]),
            int(target.data.numpy()[0]),
            round(float(output.data.numpy()[0]), 4),
            round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
        ))
