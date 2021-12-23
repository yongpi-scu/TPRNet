import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['tprnet']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeepFeatureExtractionModule(nn.Module):

    def __init__(self, block, layers, color_channels=3, feature_size=27, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(DeepFeatureExtractionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(color_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        ###########################
        # extract features from I_r
        self.conv2 = nn.Conv2d(color_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(self.inplanes)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = norm_layer(64)
        ###########################
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, feature_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self._load_pretrain()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _load_pretrain(self):
        exclusions = ['fc']
        pretrained_dict = torch.load("/root/workspace/res/ckpt/pytorch/resnet/resnet18-5c106cde.pth")
        for key in list(pretrained_dict.keys()):
            for exclusion in exclusions:
                if exclusion in key:
                    pretrained_dict.pop(key)
                    break
        pretrained_dict["conv2.weight"] = pretrained_dict["conv1.weight"]
        pretrained_dict["bn2.running_mean"] = pretrained_dict["bn1.running_mean"]
        pretrained_dict["bn2.running_var"] = pretrained_dict["bn1.running_var"]
        pretrained_dict["bn2.weight"] = pretrained_dict["bn1.weight"]
        pretrained_dict["bn2.bias"] = pretrained_dict["bn1.bias"]
        model_dict = self.state_dict()
        model_dict.update(pretrained_dict)
        print("checkpoint successeful loaded.")
        self.load_state_dict(model_dict)

    def _forward_impl(self, I_o, I_r):
        F_o = self.conv1(I_o)
        F_o = self.bn1(F_o)
        F_o = self.relu(F_o)
        F_o = self.maxpool(F_o)
        ###############
        F_r = self.conv2(I_r)
        F_r = self.bn2(F_r)
        F_r = self.relu(F_r)
        F_r = self.maxpool(F_r)

        F_r = self.conv3(F_r)
        F_r = self.bn3(F_r)
        ###############
        F_d = F_o + F_r
        F_d = self.relu(F_d)
        F_d = self.conv4(F_d)
        F_d = self.bn4(F_d)
        F_d = self.relu(F_d)
        ###############
        F_d = self.layer1(F_d)
        F_d = self.layer2(F_d)
        F_d = self.layer3(F_d)
        F_d = self.layer4(F_d)
        F_d = self.avgpool(F_d)
        F_d = torch.flatten(F_d, 1)
        F_d = self.fc(F_d)     
        return F_d

    def forward(self, I_o, I_r):
        return self._forward_impl(I_o, I_r)

class HandcraftedFeatureExtractionModule(nn.Module):

    def __init__(self, feature_size=27):
        super(HandcraftedFeatureExtractionModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feature_size, 120),
            nn.Sigmoid(),
            nn.Linear(120, feature_size),
        )
    def forward(self, V_h):
        F_h = self.fc1(V_h)
        F_h = self.fc2(F_h) + F_h
        return F_h

class ClassificationModule(nn.Module):

    def __init__(self, feature_size=54, num_classes=6):
        super(ClassificationModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Sigmoid(),
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, F_c):
        x = self.fc(F_c)
        return x

class TPRNet(nn.Module):

    def __init__(self, color_channels=1, feature_size=27, num_classes=6):
        super(TPRNet, self).__init__()
        self.deep_feature_extraction_module = DeepFeatureExtractionModule(BasicBlock, [2, 2, 2, 2], color_channels, feature_size)
        self.handcrafted_feature_extraction_module = HandcraftedFeatureExtractionModule(feature_size)
        self.classification_module = ClassificationModule(feature_size*2, num_classes)
    
    def forward(self, I_o, I_r, V_h):
        F_d = self.deep_feature_extraction_module(I_o, I_r)
        F_h = self.handcrafted_feature_extraction_module(V_h)
        x = self.classification_module(torch.cat([F_d, F_h], dim=1))
        return x

def tprnet(color_channels=3, feature_size=27, num_classes=6):
    return TPRNet(color_channels, feature_size, num_classes)










