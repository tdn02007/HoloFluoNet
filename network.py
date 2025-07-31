import torch
import torch.nn as nn
import torch.nn.functional as F
from network_cbam import CBAM
from network_aspp import ASPP

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.final = nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.bn = nn.InstanceNorm2d(out_features)

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.bn(self.final(x + self.conv_block(x)))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channel, out_channel, num_blocks = [2,2,2,2], is_distance=True, is_boundary=True, aspp=False, cbam=False):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.is_distance = is_distance
        self.is_boundary = is_boundary
        self.aspp = aspp
        self.cbam = cbam

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(Bottleneck,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # ASPP laysers
        if self.aspp:
            self.aspplayer1 = ASPP()
            self.aspplayer2 = ASPP()
            self.aspplayer3 = ASPP()
            self.aspplayer4 = ASPP()
            self.aspplayer5 = ASPP()

        if self.cbam:
            #CBAM
            CBAM_layer1 = []
            CBAM_layer2 = []
            CBAM_layer3 = []
            CBAM_layer4 = []
            CBAM_layer5 = []

            for _ in range(2):
                CBAM_layer1 += [CBAM(256, 256)]
                CBAM_layer2 += [CBAM(256, 256)]
                CBAM_layer3 += [CBAM(256, 256)]
                CBAM_layer4 += [CBAM(256, 256)]
                CBAM_layer5 += [CBAM(256, 256)]

            self.CBAM_block1 = nn.Sequential(*CBAM_layer1)
            self.CBAM_block2 = nn.Sequential(*CBAM_layer2)
            self.CBAM_block3 = nn.Sequential(*CBAM_layer3)
            self.CBAM_block4 = nn.Sequential(*CBAM_layer4)
            self.CBAM_block5 = nn.Sequential(*CBAM_layer5)

        # Upsampling
        self.Up_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Up_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Up_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Up_block4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.is_distance:
            self.Up_block5 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
            )


        # Smooth layers
        self.smooth1 = ResidualBlock(512, 256)
        self.smooth2 = ResidualBlock(512, 256)
        self.smooth3 = ResidualBlock(512, 256)

        self.final1 = nn.Conv2d(64, out_channel, kernel_size=1)
        self.final2 = nn.Conv2d(64, 1, kernel_size=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        if self.is_distance:
            # distance
            h1 = p2
            if self.aspp:
                h1 = self.aspplayer5(h1)
            if self.cbam:
                h1 = self.CBAM_block5(h1)
            h1 = self.Up_block5(h1)
            h1 = self.final2(h1)

        # ASPP
        if self.aspp:
            p5 = self.aspplayer1(p5)
            p4 = self.aspplayer2(p4)
            p3 = self.aspplayer3(p3)
            p2 = self.aspplayer4(p2)
        
        # CBAM
        if self.cbam:
            p5 = self.CBAM_block1(p5)
            p4 = self.CBAM_block2(p4)
            p3 = self.CBAM_block3(p3)
            p2 = self.CBAM_block4(p2)

        # upsampling
        p5 = self.Up_block1(p5)
        p4 = torch.cat((p5, p4), dim=1)
        p4 = self.smooth1(p4)

        p4 = self.Up_block2(p4)
        p3 = torch.cat((p4, p3), dim=1)
        p3 = self.smooth2(p3)

        p3 = self.Up_block3(p3)
        p2 = torch.cat((p3, p2), dim=1)
        p2 = self.smooth3(p2)

        p1 = self.Up_block4(p2)
        p1 = self.final1(p1)

        if self.is_distance:
            return p1, h1
        else:
            return p1
    

class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()

        main = [
            nn.Conv2d(in_channel, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
        ]
        main += [nn.InstanceNorm2d(128)]

        main += [nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, 4, 2, 1)]
        main += [nn.InstanceNorm2d(256)]

        main += [nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, 4, 1, 1)]
        main += [nn.InstanceNorm2d(512)]

        main += [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid(),
        ]

        self.main = nn.Sequential(*main)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        return self.main(out)
