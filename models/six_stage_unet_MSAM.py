import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

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


class MSAM(nn.Module):  # Multi-scale Aggregation Module

    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[3, 7]):
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
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl, data_format='channels_first'),
            nn.Conv2d(dim_xl, dim_xl, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl, data_format='channels_first'),
            nn.Conv2d(dim_xl, dim_xl, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl, 1)
        )

    def forward(self, xh, xl):

        gh = self.w_h(xh)
        gl = self.w_l(xl)
        psi_h = self.psi_h(gh)
        psi_l = self.psi_l(gl)
        xh = gh * psi_h
        xl = gl * psi_l
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        # print(torch.cat((xh[0], xl[0], xh[2], xl[2]), dim=1).size())
        x0 = self.g0(torch.cat((xh[0], xl[0], xh[2], xl[2]), dim=1))  # in_ch = xl, out_ch = xl
        x1 = self.g1(torch.cat((xh[1], xl[1], xh[3], xl[3]), dim=1))  # in_ch = xl, out_ch = xl
        x = torch.cat((x0, x1), dim=1) # c = 2 * xl
        x = self.tail_conv(x)
        return x

class SIXUNET_MSAM(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, 1, 1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, 1, 1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, 1, 1)
        self.encoder4 = nn.Conv2d(c_list[2], c_list[3], 3, 1, 1)
        self.encoder5 = nn.Conv2d(c_list[3], c_list[4], 3, 1, 1)
        self.encoder6 = nn.Conv2d(c_list[4], c_list[5], 3, 1, 1)

        if bridge:
            self.GAB1 = MSAM(c_list[1], c_list[0])
            self.GAB2 = MSAM(c_list[2], c_list[1])
            self.GAB3 = MSAM(c_list[3], c_list[2])
            self.GAB4 = MSAM(c_list[4], c_list[3])
            self.GAB5 = MSAM(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')

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
        t1 = out  # b, c0, H/2, W/2


        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4


        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8


        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16


        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32


        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32

        t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16

        t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8

        t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4

        t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2

        t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W


        return torch.sigmoid(out0)


c = 3
h = 256
w = 256
b = 16
x = torch.randn([b, c, h, w])
model = SIXUNET_MSAM(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True)
x1 = model(x)
print(x1.size())