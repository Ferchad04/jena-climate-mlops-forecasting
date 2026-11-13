import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLu(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLu()
        )
    def fordward(self, x):
        return self.block(x)

    class UNet(nn.Module):
        def __init__(self, num_classes=21):
            super(UNet, self).__init__()

            self.encoder1 = ConvBlock(3,32)
            self.encoder2 = ConvBlock(32,64)
            self.pool = nn.MaxPool2d(2,2)

            self.latent = ConvBlock(64, 128)

            self.up2 = nn.ConvTranspose2d(128,64,2,2)
            self.decoder1 = ConvBlock(64,32)

            self.output = nn.Conv2d(
                32,num_classes,1
            )

        def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            l = self.latent(self.pool(e2))

            d2 = self.up2(l)
            d2 = torch.cat((e2,d2),1)
            d2 = self.decoder2(d2)

            d1 = self.up1(l)
            d1 = torch.cat((e1,d1),1)
            d1 = self.decoder1(d1)

            return self.output(d1)
