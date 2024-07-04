## ResNet Blocks and SE Blocks

import torch.nn as nn
import torch
import torch.nn.functional as F

import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                     
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print('se reduction: ', reduction)
        # print(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: B*C*D*T
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottle2neck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(SEBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=8)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        #print('x: ', x.size())
        out = self.conv1(x)
        #print('conv1: ', out.size())
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        #print('conv2: ', out.size())
        out = self.conv3(out)
        #print('conv3: ', out.size())
        out = self.bn3(out)
        out = self.se(out)
        #print('se :', out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=2, loss='softmax', **kwargs):
        self.inplanes = 16
        super(Res2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False), 
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))


        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, layers[0])#64
        self.layer2 = self._make_layer(block, 8, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block,32, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Linear(16*block.expansion, num_classes)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale))

        return nn.Sequential(*layers)

    def _forward(self, x):
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)

        return F.log_softmax(x, dim=-1)

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward

''' ResNet models'''






def se_res2net50_1(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model


