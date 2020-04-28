# General utility functions
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import time
import math

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dice import *

def get_gaussian_2d_heatmap(num_rows, num_cols, sigma, peak_row=None, peak_col=None):
    if peak_row is None:
        peak_row = num_rows // 2
    if peak_col is None:
        peak_col = num_cols // 2
    
    (Y,X) = torch.meshgrid(torch.arange(0,num_rows), torch.arange(0,num_cols))
    
    Y = Y.float()
    X = X.float()

    return torch.exp(((X - peak_col).pow(2) + (Y - peak_row).pow(2)) / (sigma * sigma * -2)) / (2 * math.pi * sigma * sigma)

def write_floats_to_txt(file_path, floats):
    with open(file_path,'w') as out:
        for f in floats:
            out.write('{:.6f}\n'.format(f))
        out.flush()

def read_floats_from_txt(file_path):
    return torch.Tensor([float(l.strip()) for l in open(file_path).readlines()])

class RunningFloatWriter:
    def __init__(self, file_path, new_file=True):
        super(RunningFloatWriter,self).__init__()

        write_mode = 'w'
        if not new_file:
            write_mode = 'a'

        self.out = open(file_path, write_mode)

    def write(self, x):
        self.out.write('{:.6f}\n'.format(x))
        self.out.flush()

    def close(self):
        if self.out:
            self.out.flush()
            self.out.close()
            self.out = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


def center_crop(img, dst_shape):
    src_nr = img.shape[-2]
    src_nc = img.shape[-1]

    dst_nr = dst_shape[-2]
    dst_nc = dst_shape[-1]
    
    if (dst_nr != src_nr) or (dst_nc != src_nc):
        src_start_r = int((src_nr - dst_nr) / 2)
        src_end_r   = src_start_r + dst_nr
        
        src_start_c = int((src_nc - dst_nc) / 2)
        src_end_c   = src_start_c + dst_nc
        
        if img.dim() == 4:
            return img[:,:,src_start_r:src_end_r,src_start_c:src_end_c]
        elif img.dim() == 3:
            return img[:,src_start_r:src_end_r,src_start_c:src_end_c]
        else:
            assert(img.dim() == 2)
            return img[src_start_r:src_end_r,src_start_c:src_end_c]
    else:
        return img

def test_dataset(ds, net, dev=None, num_lands=0):
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        net.eval()

        losses = torch.zeros(len(ds))
        
        num_items = 0
    
        if num_lands > 0:
            criterion = DiceAndHeatMapLoss2D(skip_bg=False)
        else:
            criterion = DiceLoss2D(skip_bg=False)

        for (i, data) in enumerate(dl, 0):
            (projs, masks, lands, heats) = data
            
            if dev is not None:
                projs = projs.to(dev)
                masks = masks.to(dev)
                if num_lands > 0:
                    if len(heats.shape) > 4:
                        assert(len(heats.shape) == 5)
                        assert(heats.shape[2] == 1)
                        heats = heats.view(heats.shape[0], heats.shape[1], heats.shape[3], heats.shape[4])
                    heats = heats.to(dev)

            net_out = net(projs)
            if (num_lands > 0) or (type(net_out) is tuple):
                pred_masks     = net_out[0]
                pred_heat_maps = net_out[1]
            else:
                pred_masks = net_out

            pred_masks = center_crop(pred_masks, masks.shape)
            
            if num_lands > 0:
                pred_heat_maps = center_crop(pred_heat_maps, heats.shape)
                loss = criterion((pred_masks, pred_heat_maps), (masks, heats))
            else:
                loss = criterion(pred_masks, masks)

            losses[i] = loss.item()

            num_items += 1
        
        assert(num_items == len(ds))

        return (torch.mean(losses), torch.std(losses))

def test_dataset_ensemble(ds, nets, dev=None, num_lands=0, dice_only=False):
    num_nets = len(nets)

    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for net in nets:
            net.eval()

        losses = torch.zeros(len(ds))
        
        num_items = 0
    
        if not dice_only and (num_lands > 0):
            criterion = DiceAndHeatMapLoss2D(skip_bg=False)
        else:
            criterion = DiceLoss2D(skip_bg=False)

        for (i, data) in enumerate(dl, 0):
            (projs, masks, lands, heats) = data
            
            if dev is not None:
                projs = projs.to(dev)
                masks = masks.to(dev)
                if num_lands > 0:
                    if len(heats.shape) > 4:
                        assert(len(heats.shape) == 5)
                        assert(heats.shape[2] == 1)
                        heats = heats.view(heats.shape[0], heats.shape[1], heats.shape[3], heats.shape[4])
                    heats = heats.to(dev)

            avg_masks = None
            avg_heats = None

            for net in nets:
                net_out = net(projs)
                if (num_lands > 0) or (type(net_out) is tuple):
                    pred_masks     = net_out[0]
                    pred_heat_maps = net_out[1]
                else:
                    pred_masks = net_out

                pred_masks = center_crop(pred_masks, masks.shape)

                if avg_masks is None:
                    avg_masks = pred_masks
                else:
                    avg_masks += pred_masks
            
                if num_lands > 0:
                    pred_heat_maps = center_crop(pred_heat_maps, heats.shape)

                    if avg_heats is None:
                        avg_heats = pred_heat_maps
                    else:
                        avg_heats += pred_heat_maps
            # end for net
            
            avg_masks /= num_nets

            if num_lands > 0:
                avg_heats /= num_nets

            if not dice_only and (num_lands > 0):
                loss = criterion((avg_masks, avg_heats), (masks, heats))
            else:
                loss = criterion(avg_masks, masks)

            losses[i] = loss.item()

            num_items += 1
        
        assert(num_items == len(ds))

        return (torch.mean(losses), torch.std(losses))

def seg_dataset(ds, net, h5_f, dev=None, num_lands=0):
    orig_img_shape = ds.rob_orig_img_shape
    
    dl = DataLoader(ds, batch_size=1, shuffle=False)
   
    dst_ds = h5_f.create_dataset('nn-segs', (len(ds), *orig_img_shape),
                                 dtype='u1',
                                 chunks=(1, *orig_img_shape),
                                 compression="gzip", compression_opts=9)
    
    dst_heats_ds = None

    if num_lands > 0:
        dst_heats_ds = h5_f.create_dataset('nn-heats', (len(ds), num_lands, *orig_img_shape),
                                           chunks=(1,1,*orig_img_shape),
                                           compression="gzip", compression_opts=9)

    with torch.no_grad():
        net.eval()

        num_items = 0
        
        for (i, data) in enumerate(dl, 0):
            projs = data[0]

            if dev is not None:
                projs = projs.to(dev)

            net_out = net(projs)
            if (num_lands > 0) or (type(net_out) is tuple):
                pred_masks = net_out[0]
                pred_heats = net_out[1]
            else:
                pred_masks = net_out

            pred_masks = center_crop(pred_masks, orig_img_shape)

            (_, pred_masks) = torch.max(pred_masks, dim=1)
   
            # write to file
            dst_ds[i,:,:] = pred_masks.view(orig_img_shape).cpu().numpy()

            if dst_heats_ds is not None:
                dst_heats_ds[i,:,:,:] = center_crop(pred_heats, orig_img_shape).numpy()

            num_items += 1
        
        assert(num_items == len(ds))


def seg_dataset_ensemble(ds, nets, h5_f, dev=None, num_lands=0, times=None, multi_class_labels=False):
    num_nets = len(nets)

    orig_img_shape = ds.rob_orig_img_shape
    
    dl = DataLoader(ds, batch_size=1, shuffle=False)
 
    dst_ds = None
    if not multi_class_labels:
        dst_ds = h5_f.create_dataset('nn-segs', (len(ds), *orig_img_shape),
                                     dtype='u1',
                                     chunks=(1, *orig_img_shape),
                                     compression="gzip", compression_opts=9)
    
    dst_heats_ds = None

    if num_lands > 0:
        dst_heats_ds = h5_f.create_dataset('nn-heats', (len(ds), num_lands, *orig_img_shape),
                                           chunks=(1,1,*orig_img_shape),
                                           compression="gzip", compression_opts=9)

    with torch.no_grad():
        for net in nets:
            net.eval()

        num_items = 0
        
        for (i, data) in enumerate(dl, 0):
            projs = data[0]
            
            start_time = time.time()

            if dev is not None:
                projs = projs.to(dev)

            avg_masks = None
            
            avg_heats = None

            for net in nets:
                net_out = net(projs)
                if (num_lands > 0) or (type(net_out) is tuple):
                    pred_masks = net_out[0]
                    pred_heats = net_out[1]
                else:
                    pred_masks = net_out

                pred_masks = center_crop(pred_masks, orig_img_shape)

                if avg_masks is None:
                    avg_masks = pred_masks
                else:
                    avg_masks += pred_masks
            
                if dst_heats_ds is not None:
                    pred_heats = center_crop(pred_heats, orig_img_shape)
                    
                    for land_idx in range(pred_heats.shape[1]):
                        pred_heats_min = pred_heats[0,land_idx,:,:].min().item()
                        pred_heats_max = pred_heats[0,land_idx,:,:].max().item()
                        #pred_heats[0,land_idx,:,:] = (pred_heats[1,land_idx,:,:] - pred_heats_min) / (pred_heats_max - pred_heats_min)
                        
                        pred_heats[0,land_idx,:,:] -= pred_heats_min

                    if avg_heats is None:
                        avg_heats = pred_heats
                    else:
                        avg_heats += pred_heats
            
            # technically we don't need to do this for the segmentation
            avg_masks /= num_nets

            if multi_class_labels:
                pred_masks = avg_masks
            else:
                (_, pred_masks) = torch.max(avg_masks, dim=1)
            
            stop_time = time.time()

            if times is not None:
                times.append(stop_time - start_time)
            
            if multi_class_labels and (dst_ds is None):
                dst_ds = h5_f.create_dataset('nn-segs', (len(ds), avg_masks.shape[1], *orig_img_shape),
                                             chunks=(1, 1, *orig_img_shape),
                                             compression="gzip", compression_opts=9)

            # write to file
            if multi_class_labels:
                dst_ds[i,:,:,:] = pred_masks.view((pred_masks.shape[1], *orig_img_shape)).cpu().numpy()
            else:
                dst_ds[i,:,:] = pred_masks.view(orig_img_shape).cpu().numpy()
            
            if dst_heats_ds is not None:
                avg_heats /= num_nets
                dst_heats_ds[i,:,:,:] = avg_heats.cpu().numpy()

            num_items += 1
        
        assert(num_items == len(ds))


