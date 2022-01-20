import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv block with
    conv2d
    pooling
    relu
    """

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel,
            padding=1,
            bias=False,
            with_pooling=True,
            pooling_fn='max',
            pooling_kernel=2,
            pooling_stride=2,
            pooling_pad=0,
    ):
        super(ConvBlock, self).__init__()
        self.with_pooling = with_pooling
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=bias)
        if with_pooling:
            if pooling_fn == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pooling_kernel, stride=pooling_stride,
                                            padding=pooling_pad)
            elif pooling_fn == 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel, stride=pooling_stride,
                                            padding=pooling_pad)
            else:
                raise Exception('unsupported pooling function: %s' % (pooling_fn,))
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv2d(x)
        if self.with_pooling:
            y = self.pooling(y)
        y = self.relu(y)
        return y


class CnnNet1(nn.Module):
    """nromal cnn net"""

    def __init__(self):
        super(CnnNet1, self).__init__()
        self.conv1 = ConvBlock(1, 10, 5, padding=0)
        self.conv2 = ConvBlock(10, 20, 5, padding=0)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.flat(y)
        y = self.fc(y)
        return y


class CnnNet2(nn.Module):
    """Another CNN net"""

    def __init__(self):
        super(CnnNet2, self).__init__()
        self.conv1 = ConvBlock(1, 16, 3, padding=0, bias=0)
        self.conv2 = ConvBlock(16, 32, 3, padding=1, bias=0)
        self.conv3 = ConvBlock(32, 64, 3, padding=1, bias=0)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(9 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.flat(y)
        y = self.fc1(y)
        y = self.fc2(y)
        return y


class InceptionBlock(nn.Module):
    """Inception Block"""

    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch_3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        # branch 1 -- avg pool
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)
        # branch 2 -- 1x1
        branch_1x1 = self.branch_1x1(x)
        # branch 3 -- 5x5
        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)
        # branch 4 -- 3x3
        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        # concat
        y = torch.cat([branch_pool, branch_1x1, branch_5x5, branch_3x3], dim=1)
        return y


class GoogleNet(nn.Module):
    """Inception net
    可以综合不同卷积核所提取的特征向量
    """

    def __init__(self):
        super(GoogleNet, self).__init__()
        # 1,28,28
        self.conv1 = ConvBlock(1, 10, kernel=5, padding=0, pooling_stride=2)
        # 10,12,12
        self.incep1 = InceptionBlock(10)
        # 88,12,12
        self.conv2 = ConvBlock(88, 20, kernel=5, padding=0, pooling_stride=2)
        # 20,4,4
        self.incep2 = InceptionBlock(20)
        # 88,4,4
        self.flat = nn.Flatten()
        # 1408
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.incep1(x)
        x = self.conv2(x)
        x = self.incep2(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.channels = in_channels
        self.conv_b = ConvBlock(self.channels, self.channels, kernel=3, padding=1, with_pooling=False)
        # self.conv2 = ConvBlock(self.channels, self.channels, kernel=3, padding=1, with_pooling=False)
        self.conv2d = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv_b(x)
        y = self.conv2d(y)
        return F.relu(x + y)


class ResidualNet(nn.Module):
    """Residual Net
    神经网络层数增加，可能导致梯度消失
    在每一个卷积块的计算结果y中加上输入值x,保证梯度能够得到有效更新
    """

    def __init__(self):
        super(ResidualNet, self).__init__()
        # 1,28,28
        self.conv_b1 = ConvBlock(1, 16, kernel=5, padding=0)
        # 16,12,12
        self.rblock1 = ResidualBlock(16)
        # 16,12,12
        self.conv_b2 = ConvBlock(16, 32, kernel=5, padding=0)
        # 32,4,4
        self.rblock2 = ResidualBlock(32)
        # 32,4,4
        self.flatten = nn.Flatten()
        # 512
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_b1(x)
        x = self.rblock1(x)
        x = self.conv_b2(x)
        x = self.rblock2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
