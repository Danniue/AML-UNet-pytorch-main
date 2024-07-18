import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

class CRM(nn.Module): # Channel Reconstruct Module

    def __init__(self, in_channels, alpha=0.1, squeeze_radio=2, group_size=2, group_kernel_size=3):

        super().__init__()

        self.up_channel = up_channel = int(alpha * in_channels,)  # up
        self.low_channel = low_channel = in_channels - up_channel  # down
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # up
        self.GWC = nn.Conv2d(up_channel//squeeze_radio, in_channels, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel//squeeze_radio, in_channels, kernel_size=1, bias=False)

        # low
        self.PWC2 = nn.Conv2d(low_channel//squeeze_radio, in_channels-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class CRSAM(nn.Module):  # Channel Reconstrution and Spatial Attention Module

    def __init__(self,
                 in_channels,
                 out_channels,
                 rate=4,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,):
        super(CRSAM, self).__init__()

        self.channel_attention = CRM(in_channels,
                                     alpha=alpha,
                                     squeeze_radio=squeeze_radio,
                                     group_size=group_size,
                                     group_kernel_size=group_kernel_size)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        x_channel_att = self.channel_attention(x).sigmoid()
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out

class DepthWiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Connect(nn.Module):  # Multi-scale Aggregation Module

    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.w_h = nn.Sequential(
            nn.Conv2d(dim_xh, dim_xh, 1, 1, 0),
            nn.BatchNorm2d(dim_xh)
        )
        self.w_l = nn.Sequential(
            nn.Conv2d(dim_xl, dim_xl, 1, 1, 0),
            nn.BatchNorm2d(dim_xl)
        )
        self.relu = nn.LeakyReLU()
        self.psi_h = nn.Sequential(
            nn.Conv2d(dim_xh, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.psi_l = nn.Sequential(
            nn.Conv2d(dim_xl, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)

        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl, 1)
        )

    def forward(self, xh, xl):

        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        x = torch.cat((xh, xl), dim=1) # c = 2 * xl
        x = self.tail_conv(x)
        return x

class SIXUNET_CRSAM(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64]):
        super().__init__()

        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, 1, 1)
        self.a1 = CRSAM(in_channels=c_list[0], out_channels=c_list[0])
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, 1, 1)
        self.a2 = CRSAM(in_channels=c_list[1], out_channels=c_list[1])
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, 1, 1)
        self.a3 = CRSAM(in_channels=c_list[2], out_channels=c_list[2])
        self.encoder4 = nn.Conv2d(c_list[2], c_list[3], 3, 1, 1)
        self.a4 = CRSAM(in_channels=c_list[3], out_channels=c_list[3])
        self.encoder5 = nn.Conv2d(c_list[3], c_list[4], 3, 1, 1)
        self.a5 = CRSAM(in_channels=c_list[4], out_channels=c_list[4])
        self.encoder6 = nn.Conv2d(c_list[4], c_list[5], 3, 1, 1)


        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))

        out = self.a1(out)
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))

        out = self.a2(out)
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))

        out = self.a3(out)
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))

        out = self.a4(out)
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))

        out = self.a5(out)
        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32

        out = F.gelu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16


        out = F.gelu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8


        out = F.gelu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4




        out = F.gelu(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2



        out = F.interpolate(self.final(out), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W
        return torch.sigmoid(out)


# c = 3
# h = 256
# w = 256
# b = 16
# x = torch.randn([b, c, h, w])
# model = SIXUNET_CRSAM(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64])
# x1 = model(x)
# print(x1.size())