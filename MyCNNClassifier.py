import torch.nn as nn
import torch.nn.functional as F


# Одно входовая Сеть
# Convolutional neural network (two convolutional layers)
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, action_size):
        super().__init__()
        # conv2d - 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # conv2d - 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # liniar - 1
        self.fc1 = nn.Linear(1024, 512)
        # dropout - 1
        self.drop_out = nn.Dropout(0.25)
        # liniar - 2
        self.fc2 = nn.Linear(512, 256)
        # liniar - 3
        self.fc3 = nn.Linear(256, action_size)
        self.softmax = nn.Softmax(dim=1)

        # conv2d - 1
        self.end1_out = nn.Linear(4096, action_size)
        self.end1_softmax = nn.Softmax(dim=1)

        # conv2d - 2
        self.end2_out = nn.Linear(1024, action_size)
        self.end2_softmax = nn.Softmax(dim=1)

        # liniar - 1
        self.end3_out = nn.Linear(512, action_size)
        self.end3_softmax = nn.Softmax(dim=1)

        # dropout - 1
        self.end4_out = nn.Linear(512, action_size)
        self.end4_softmax = nn.Softmax(dim=1)

    def forward1(self, x_0):
        out = self.layer1(x_0)
        out = out.reshape(out.size(0), -1)
        out = self.end1_out(out)
        out = self.end1_softmax(out)
        return out

    def forward2(self, x_0):
        out = self.layer1(x_0)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.end2_out(out)
        out = self.end2_softmax(out)
        return out

    def forward3(self, x_0):
        out_0 = self.layer1(x_0)
        out_0 = self.layer2(out_0)
        out_0 = out_0.reshape(out_0.size(0), -1)

        out = F.relu(self.fc1(out_0))
        out = self.end3_out(out)
        out = self.end3_softmax(out)
        return out

    def forward4(self, x_0):
        out_0 = self.layer1(x_0)
        out_0 = self.layer2(out_0)
        out_0 = out_0.reshape(out_0.size(0), -1)

        out = F.relu(self.fc1(out_0))
        out = self.drop_out(out)
        out = self.end4_out(out)
        out = self.end4_softmax(out)
        return out

    def forward(self, x_0):
        out_0 = self.layer1(x_0)
        out_0 = self.layer2(out_0)
        out_0 = out_0.reshape(out_0.size(0), -1)

        out = F.relu(self.fc1(out_0))
        out = self.drop_out(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out