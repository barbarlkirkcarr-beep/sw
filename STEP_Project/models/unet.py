import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()
        
        # 辅助函数：卷积块 (Conv-ReLU-Conv-ReLU)
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Encoder (根据论文描述的层级)
        self.enc1 = double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = double_conv(256, 512) # Bottleneck
        
        # Decoder (上采样 + Skip Connection)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = double_conv(512, 256) # 256 + 256
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = double_conv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = double_conv(64, 32)
        
        # Output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channels, H, W]
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.enc5(self.pool4(e4))
        
        # Decoding
        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)
        
        out = self.sigmoid(self.final_conv(d4))
        return out # Probability map [0, 1]