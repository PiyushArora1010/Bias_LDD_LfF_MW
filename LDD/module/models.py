from cgi import print_directory
from json import encoder
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.memory import MemoryWrapLayer, BaselineMemory
from module.attention import AttentionLayer
from torch.autograd import Variable
from entmax import sparsemax
from torchvision import models
import torch.nn.init as init

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)

        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        final_out = self.fc(out)
      
        return final_out

class ResNet_MW(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_MW, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = MemoryWrapLayer(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)

        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction
    def forward_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self,x,memory_input, return_weights = False):
        x_out = self.forward_encoder(x)
        mem_out = self.forward_encoder(memory_input)
        out_mw = self.fc(x_out,mem_out, return_weights)
        return out_mw

class ResNet_attention(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_attention, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = AttentionLayer(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)

        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction
    def forward_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self,x,memory_input, return_weights = False):
        x_out = self.forward_encoder(x)
        mem_out = self.forward_encoder(memory_input)
        out_mw = self.fc(x_out,mem_out, return_weights)
        return out_mw

class ResNet_sim(nn.Module):
    def __init__(self, model_name, num_classes = 10, prev_dim = 256):
        super(ResNet_sim, self).__init__()
        if model_name == 'resnet18_sim':
            self.model = models.resnet18(pretrained = False,)
        elif model_name == 'resnet50_sim':
            self.model = models.resnet50(pretrained = False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.predictor = nn.Sequential(
            nn.Linear(1024, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1024, bias = False)
        )
    def forward(self, x):
        x = self.model(x)
        return x
    def _predictor(self, x):
        x = self.predictor(x)
        return x
    
def resnet20(num_classes = 10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def resnet32(num_classes = 10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)

def resnet44(num_classes = 10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)

def resnet20_MW(num_classes = 10):
    return ResNet_MW(BasicBlock, [3, 3, 3], num_classes)

def resnet32_MW(num_classes = 10):
    return ResNet_MW(BasicBlock, [5, 5, 5], num_classes)

def resnet44_MW(num_classes = 10):
    return ResNet_MW(BasicBlock, [7, 7, 7], num_classes)

def resnet20_attention(num_classes = 10):
    return ResNet_attention(BasicBlock, [3, 3, 3], num_classes)

def resnet32_attention(num_classes = 10):
    return ResNet_attention(BasicBlock, [5, 5, 5], num_classes)

def resnet44_attention(num_classes = 10):
    return ResNet_attention(BasicBlock, [7, 7, 7], num_classes)

dic_models = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet20_MW': resnet20_MW,
    'resnet32_MW': resnet32_MW,
    'resnet44_MW': resnet44_MW,
    'resnet20_attention': resnet20_attention,
    'resnet32_attention': resnet32_attention,
    'resnet44_attention': resnet44_attention,
}