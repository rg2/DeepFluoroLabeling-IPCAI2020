# U-Net model
#
# Adapted from: https://github.com/jvanvugt/pytorch-unet,
# which was adapted from https://discuss.pytorch.org/t/unet-implementation/426
#
# MIT License
# 
# Copyright (c) 2018 Joris
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# Modifications are Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
from torch import nn
import torch.nn.functional as F

import util

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6,
                 padding=False, pad_mode='zeros',
                 batch_norm=False, up_mode='upconv', max_pool=True, num_lands=0,
                 do_res=True, block_depth=2, lands_block_depth=0, lands_num_1x1=2,
                 do_soft_max=True):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            pad_mode : 'zeros' or 'circular'
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.pad_mode = pad_mode
        self.depth = depth
        self.do_max_pool = max_pool
        self.num_lands = num_lands
        self.do_soft_max = do_soft_max

        self.downsample_convs = None
        if not self.do_max_pool:
            self.downsample_convs = nn.ModuleList()

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm, pad_mode,
                                                do_res=do_res, block_depth=block_depth))
            prev_channels = 2**(wf+i)
            
            if not self.do_max_pool:
                self.downsample_convs.append(nn.Conv2d(prev_channels, prev_channels, kernel_size=2, stride=2))

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, pad_mode, do_res=do_res, block_depth=block_depth))
            prev_channels = 2**(wf+i)

        self.seg_conv = nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=False)

        if do_soft_max:
            self.soft_max = nn.Softmax2d()

        if self.num_lands > 0:
            lands_use_non_lin = False
            
            self.lands_block = None

            lands_block_num_chan = prev_channels
            
            if lands_block_depth > 0:
                lands_block_num_chan = prev_channels // 2

                lands_block = []

                lands_block.append(nn.Conv2d(prev_channels, lands_block_num_chan,
                                             kernel_size=3,
                                             padding=int(padding), padding_mode=pad_mode))
                
                if lands_use_non_lin:
                    lands_block.append(nn.ReLU())
                    if batch_norm:
                        lands_block.append(nn.BatchNorm2d(lands_block_num_chan))
               
                for i in range(lands_block_depth-1):
                    lands_block.append(nn.Conv2d(lands_block_num_chan, lands_block_num_chan,
                                                 kernel_size=3,
                                                 padding=int(padding), padding_mode=pad_mode))
                
                    if lands_use_non_lin:
                        lands_block.append(nn.ReLU())
                        if batch_norm:
                            lands_block.append(nn.BatchNorm2d(lands_block_num_chan))

                self.lands_block = nn.Sequential(*lands_block)

            assert(lands_num_1x1 > 0)

            lands_1x1 = []
            
            lands_1x1_num_out_feat = num_lands
            
            if lands_num_1x1 > 1:
                lands_1x1_num_out_feat = num_lands + n_classes
            
            lands_1x1.append(nn.Conv2d(lands_block_num_chan + n_classes,
                                       lands_1x1_num_out_feat,
                                       kernel_size=1, bias=False))

            for i in range(lands_num_1x1 - 1):
                lands_1x1.append(nn.Conv2d(lands_1x1_num_out_feat,
                                           num_lands,
                                           kernel_size=1, bias=False))

                lands_1x1_num_out_feat = num_lands

            self.lands_1x1 = nn.Sequential(*lands_1x1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)

                if self.do_max_pool:
                    x = F.max_pool2d(x, 2)
                else:
                    x = self.downsample_convs[i](x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        seg_x = self.seg_conv(x)

        if self.do_soft_max:
            seg = self.soft_max(seg_x)
        else:
            seg = seg_x

        if self.num_lands > 0:
            if self.lands_block is not None:
                x = self.lands_block(x)
            
            x = torch.cat((x, util.center_crop(seg_x, x.shape)), dim=1)

            heat_maps = self.lands_1x1(x)
            
            return (seg, heat_maps)
        else:
            return seg


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, pad_mode, do_res, block_depth):
        super(UNetConvBlock, self).__init__()

        assert(block_depth > 0)

        # TODO: should we try doing batch norm before ReLU?

        self.do_res = do_res

        if do_res:
            self.res_conv1x1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)

        block = []
        
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding), padding_mode=pad_mode))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        for d in range(block_depth-1):
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                                   padding=int(padding), padding_mode=pad_mode))
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        out = self.block(x)
        
        if self.do_res:
            res = self.res_conv1x1(x)
            out += res

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, pad_mode, do_res, block_depth):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, pad_mode, do_res=do_res, block_depth=block_depth)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
