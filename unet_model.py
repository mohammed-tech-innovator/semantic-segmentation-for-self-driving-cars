import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up,self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 13):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.norm = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) 
        self.down4 = Down(512, 1024 )
        self.up1 = Up(1024, 512 )
        self.up2 = Up(512, 256 )
        self.up3 = Up(256, 128 )
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def train(UNET, batch, optim):
    lossFunction = nn.CrossEntropyLoss()
    x,y = batch
    x = x.float()
    y = y.long()
    x = x.to(device)
    y = y.to(device)
    UNET.zero_grad()
    output = UNET.forward(x)
    loss = lossFunction(output , y)
    loss.backward()
    optim.step()

    del x
    del y 
    del lossFunction

    return UNET,loss


def test(UNET,  batch):
    lossFunction = nn.CrossEntropyLoss()
    x,y = batch
    x = x.float()
    y = y.long()
    x = x.to(device)
    y = y.to(device)
    output = UNET.forward(x)
    loss = lossFunction(output , y)
    del x
    del y 
    del lossFunction

    return loss




def show_example(UNET,batch):
    x,y = batch
    x = x.float()
    x = x.to('cpu')
    net = UNET.to('cpu')
    output = net.forward(x)
    randomIndex = np.random.randint(0,len(output))
    image = x.cpu().detach().numpy()[randomIndex]
    result = torch.argmax(output[randomIndex],dim=0)
    image = np.reshape(image , (image.shape[1],image.shape[2],image.shape[0]))
    f, axarr = plt.subplots(1,2,figsize=(15,3))
    axarr[0].imshow(image/255)
    axarr[1].imshow(result)
    UNET.to(device)
    x.to(device)
    plt.pause(1.0)

    return 


