import torch


class SegNet(torch.nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.fc1 = torch.nn.Linear(128 * 128, 10000)
        self.act1 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(1000, 10000)
        # self.act2 = torch.nn.ReLU()

        # self.fc3 = torch.nn.Linear(10000, 1000)
        # self.act3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(10000, 128 * 128)
        self.act4 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        # x = self.fc2(x)
        # x = self.act2(x)

        # x = self.fc3(x)
        # x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        return x
