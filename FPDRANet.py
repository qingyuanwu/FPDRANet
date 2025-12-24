import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matlab.engine

# Start the MATLAB engine
eng = matlab.engine.start_matlab()

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Up(nn.Module):
    """
    Upsampling block using Transposed Convolution.
    """

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        """
        Forward pass for the Upsampling block.

        Args:
            x1: Input tensor to be upsampled.
            x: Reference tensor for target spatial dimensions (for padding).
        """
        x2 = self.up(x1)

        # Calculate padding to match dimensions of x
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]

        # Pad the upsampled tensor
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


# ------------------------------------------------------------------------------
# Spatial Attention Modules
# ------------------------------------------------------------------------------

class Basic(nn.Module):
    """
    Basic convolutional block used within the Spatial Attention mechanism.
    """

    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """
    Channel Pooling layer.
    Concatenates the max pooling and mean pooling along the channel axis.
    """

    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    """
    Spatial Attention Block (SAB).
    """

    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


# ------------------------------------------------------------------------------
# Channel Attention Modules
# ------------------------------------------------------------------------------

class CAB(nn.Module):
    """
    Channel Attention Block (CAB).
    """

    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RAB(nn.Module):
    """
    Residual Attention Block (RAB).
    Combines Residual Blocks with Spatial Attention.
    """

    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        # Cascaded residual connections
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        # Apply spatial attention
        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    """

    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class HDRAB(nn.Module):
    """
    High-Density Residual Attention Block (HDRAB).
    A complex block integrating multi-scale convolutions, SE blocks, and CAB.
    """

    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8

        self.cab = CAB(in_channels, reduction, bias)

        # Standard convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        # Dilated convolutions for larger receptive field
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        # Reverse path with dilated convolutions
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        # Tail convolution
        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

        self.se_block = SEBlock(in_channels, out_channels)

    def forward(self, y):
        # Forward pass through dilated convolutions and SE blocks with residual connections
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2 = self.se_block(y2)
        y2_1 = y2 + y

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4 = self.se_block(y4)
        y4_1 = y4 + y2_1

        # Backward/Integration pass
        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1 + y3)
        y6 = self.se_block(y6)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)

        y8 = self.conv_tail(y7_1)
        y8 = self.se_block(y8)
        y8_1 = y8 + y6_1

        # Final Channel Attention
        y9 = self.cab(y8_1)
        y9_1 = y + y9

        return y9_1


class FPDRANet(nn.Module):
    """
    Dual Residual Attention Network (DRANet).
    Consists of two branches (upper and lower) handling different inputs,
    fusing them at the end.
    """

    def __init__(self, in_nc=2, out_nc=1, nc=128, bias=True):
        super(FPDRANet, self).__init__()
        kernel_size = 3

        # Upper branch input head
        self.conv_head_upper = nn.Conv2d(1, nc, kernel_size=kernel_size, padding=1, bias=bias)

        # Lower branch input head (expecting high channel count input)
        self.conv_head_lower = nn.Conv2d(2000, nc, kernel_size=kernel_size, padding=1, bias=bias)

        # Attention blocks
        self.rab = RAB(nc, nc, bias)
        self.hdrab = HDRAB(nc, nc, bias)

        # Tail convolutions
        self.conv_tail = nn.Conv2d(nc, 150, kernel_size=kernel_size, padding=1, bias=bias)
        self.dual_tail = nn.Conv2d(2 * 150, 150, kernel_size=kernel_size, padding=1, bias=bias)

        # Downsampling and Upsampling
        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)
        self.up = Up(nc, bias)

        self.se_block = SEBlock(nc, nc)

    def forward(self, x, y):
        # ------------------------------------
        # Upper Branch Processing (UNet-like structure with RAB)
        # ------------------------------------
        x1 = self.conv_head_upper(x)
        x2 = self.rab(x1)
        x2_1 = self.down(x2)
        x3 = self.rab(x2_1)
        x3_1 = self.down(x3)
        x4 = self.rab(x3_1)

        # Upsampling path
        x4_1 = self.up(x4, x3)
        x5 = self.rab(x4_1 + x3)
        x5_1 = self.up(x5, x2)
        x6 = self.rab(x5_1 + x2)
        x7 = self.conv_tail(x6 + x1)
        X = x7

        # ------------------------------------
        # Lower Branch Processing (Cascaded HDRAB)
        # ------------------------------------
        y1 = self.conv_head_lower(y)
        y2 = self.hdrab(y1)
        y3 = self.hdrab(y2)
        y4 = self.hdrab(y3)
        y5 = self.hdrab(y4 + y3)
        y6 = self.hdrab(y5 + y2)
        y7 = self.conv_tail(y6 + y1)
        Y = y7

        # ------------------------------------
        # Fusion
        # ------------------------------------
        z1 = torch.cat([X, Y], dim=1)
        z = self.dual_tail(z1)

        return z, X, Y


if __name__ == '__main__':
    def count_params(model):
        """Calculate the total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def count_flops(model, input_shapes=[(1, 3, 224, 224), (32, 2000, 64, 64)]):
        """
        Calculate FLOPs for a multi-input model.

        Args:
            model: The PyTorch model to analyze.
            input_shapes: List of input tensor shapes, e.g., [(x_shape), (y_shape)].
        """
        from thop import profile

        # Generate dummy input tensors
        inputs = [torch.randn(shape).to(device) for shape in input_shapes]

        # Calculate FLOPs
        flops, _ = profile(model, inputs=tuple(inputs))
        return flops


    # Instantiate the model
    model = FPDRANet().to(device)

    # Calculate parameters
    params = count_params(model)

    # Define input shapes for FLOPs calculation
    input_shapes = [
        (32, 1, 64, 64),  # First input shape: (batch, channel, H, W)
        (32, 2000, 64, 64)  # Second input shape
    ]
    flops = count_flops(model, input_shapes)

    print(f"Parameters: {params / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f}G")