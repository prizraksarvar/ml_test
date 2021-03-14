import torch.nn as nn
import torch.nn.functional as F


# Одно входовая Сеть
# Convolutional neural network (two convolutional layers)
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, action_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(1024, 512)
        self.drop_out = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.softmax = nn.Softmax(dim=1)

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