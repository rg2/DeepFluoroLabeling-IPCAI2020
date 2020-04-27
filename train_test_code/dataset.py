# Dataloading utilities from preprocessed HDF5 files.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import math
import random

import h5py as h5

import torch
import torch.utils.data

import torchvision.transforms.functional as TF

import numpy as np

import PIL

from util import *
            
def calc_pad_amount(padded_img_dim, cur_img_dim):
    # new pad dimension should be larger
    assert(padded_img_dim > cur_img_dim)

    # first calculate the amount to pad along the borders
    pad = (padded_img_dim - cur_img_dim)/ 2

    # handle odd sized input
    if pad != int(pad):
        pad = int(pad) + 1
    else:
        # needs to be integral
        pad = int(pad)

    return pad

class RandomDataAugDataSet(torch.utils.data.Dataset):
    def __init__(self, projs, segs, lands=None, proj_pad_dim=0):
        self.projs = projs
        self.segs  = segs
        self.lands = lands
   
        assert(len(projs.shape) == 4)
        assert(projs.shape[1] == 1)

        if segs is not None:
            assert(len(projs.shape) == len(segs.shape))
            assert(projs.shape[0] == segs.shape[0])
            
            # initial sizes before padding should be equal
            assert(projs.shape[2] == segs.shape[2])
            assert(projs.shape[3] == segs.shape[3])
        
        if lands is not None:
            assert(projs.shape[0] == lands.shape[0])
            assert(lands.shape[1] == 2)

        self.prob_of_aug = 0.5
        #self.prob_of_aug = 1.0

        self.do_invert = True
        self.do_gamma  = True
        self.do_noise  = True
        self.do_affine = True
        self.do_erase  = True

        self.erase_prob = 0.25

        self.pad_data_for_affine = True

        self.do_norm_01_scale = True

        self.include_heat_map = True

        self.print_aug_info = False

        self.extra_pad = 0
        if proj_pad_dim > 0:
            # only support square images for now
            assert(projs.shape[-1] == projs.shape[-2])
            self.extra_pad = calc_pad_amount(proj_pad_dim, projs.shape[-1])

    def __len__(self):
        return self.projs.shape[0]

    def __getitem__(self, i):
        assert(type(i) is int)

        p = self.projs[i,:,:,:]

        s = None
        if self.segs is not None:
            cur_seg = self.segs[i,:,:,:]

        cur_lands = None
        if self.lands is not None:
            # we need a deep copy here because of possible data aug
            cur_lands = self.lands[i,:,:].clone()

        need_to_pad_proj = self.extra_pad > 0

        if (self.prob_of_aug > 0) and (random.random() < self.prob_of_aug): 
            #print('augmenting...')

            if self.do_invert and (random.random() < 0.5):
                #print('  inversion...')

                p_max = p.max()
                #p_min = p.min()
                p = p_max - p

                if self.print_aug_info:
                    print('inverting')

            if self.do_noise:
                # normalize to [0,1] to apply noise
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / (p_max - p_min)

                cur_noise_sigma = random.uniform(0.005, 0.01)
                p += torch.randn(p.shape) * cur_noise_sigma
                
                p = (p * (p_max - p_min)) + p_min

                if self.print_aug_info:
                    print('noise sigma: {:.3f}'.format(cur_noise_sigma))

            if self.do_gamma:
                # normalize to [0,1] to apply gamma
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / (p_max - p_min)

                gamma = random.uniform(0.7,1.3)
                p.pow_(gamma)

                p = (p * (p_max - p_min)) + p_min

                if self.print_aug_info:
                    print('gamma = {:.2f}'.format(gamma))
       
            if self.do_affine:
                # data needs to be in [0,1] for PIL functions
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / (p_max - p_min)
                
                orig_p_shape = p.shape
                if self.pad_data_for_affine:
                    pad1 = int(math.ceil(orig_p_shape[1] / 2.0))
                    pad2 = int(math.ceil(orig_p_shape[2] / 2.0))
                    if need_to_pad_proj:
                        pad1 += self.extra_pad
                        pad2 += self.extra_pad
                        need_to_pad_proj = False

                    p = torch.from_numpy(np.pad(p.numpy(),
                                                ((0,0), (pad1,pad1), (pad2,pad2)),
                                                'reflect'))
                
                p_il = TF.to_pil_image(p)

                # this uniformly samples the direction
                rand_trans = torch.randn(2)
                rand_trans /= rand_trans.norm()

                # now uniformly sample the magnitdue
                rand_trans *= random.random() * 20
                
                rot_ang = random.uniform(-5, 5)
                trans_x = rand_trans[0]
                trans_y = rand_trans[1]
                shear   = random.uniform(-2, 2)
                
                scale_factor = random.uniform(0.9, 1.1)

                if self.print_aug_info:
                    print('Rot: {:.2f}'.format(rot_ang))
                    print('Trans X: {:.2f} , Trans Y: {:.2f}'.format(trans_x, trans_y))
                    print('Shear: {:.2f}'.format(shear))
                    print('Scale: {:.2f}'.format(scale_factor))

                p = TF.to_tensor(TF.affine(TF.to_pil_image(p),
                                 rot_ang,
                                 (trans_x, trans_y),
                                 scale_factor,
                                 shear,
                                 resample=PIL.Image.BILINEAR))
                
                if self.pad_data_for_affine:
                    # pad can be zero
                    pad_shape = (orig_p_shape[-2] + (2 * self.extra_pad), orig_p_shape[-1] + (2 * self.extra_pad))
                    p = center_crop(p, pad_shape)

                p = (p * (p_max - p_min)) + p_min

                if cur_seg is not None:
                    orig_s_shape = cur_seg.shape
                    if self.pad_data_for_affine:
                        pad1 = int(math.ceil(orig_s_shape[1] / 2.0))
                        pad2 = int(math.ceil(orig_s_shape[2] / 2.0))
                        cur_seg = torch.from_numpy(np.pad(cur_seg.numpy(),
                                                    ((0,0), (pad1,pad1), (pad2,pad2)),
                                                    'reflect'))
                    
                    # warp each class separately, I don't want any wacky color
                    # spaces assumed by PIL
                    for c in range(cur_seg.shape[0]):
                        cur_seg[c,:,:] = TF.to_tensor(TF.affine(TF.to_pil_image(cur_seg[c,:,:]),
                                                          rot_ang,
                                                          (trans_x, trans_y),
                                                          scale_factor,
                                                          shear))
                    
                    # renormalize after warping
                    cur_seg /= torch.sum(cur_seg, dim=0)

                    if self.pad_data_for_affine:
                        cur_seg = center_crop(cur_seg, orig_s_shape)
                
                if cur_lands is not None:
                    shape_for_center_of_rot = cur_seg.shape if cur_seg is not None else p.shape

                    center_of_rot = ((shape_for_center_of_rot[-2] / 2.0) + 0.5,
                                     (shape_for_center_of_rot[-1] / 2.0) + 0.5)
                    
                    A_inv = TF._get_inverse_affine_matrix(center_of_rot, rot_ang, (trans_x, trans_y), scale_factor, shear)
                    A = np.matrix([ [A_inv[0], A_inv[1], A_inv[2]], [A_inv[3], A_inv[4], A_inv[5]], [0,0,1]]).I

                    for pt_idx in range(cur_lands.shape[-1]):
                        cur_land = cur_lands[:,pt_idx]
                        if (not math.isinf(cur_land[0])) and (not math.isinf(cur_land[1])):
                            tmp_pt = A * np.asmatrix(np.pad(cur_land.numpy(), (0,1), mode='constant', constant_values=1).reshape(3,1))
                            xform_l = torch.from_numpy(np.squeeze(np.asarray(tmp_pt))[0:2])
                            if (cur_seg is not None) and \
                               ((xform_l[0] < 0) or (xform_l[0] > (orig_s_shape[1] - 1)) or \
                                (xform_l[1] < 0) or (xform_l[1] < (orig_s_shape[0] - 1))):
                                xform_l[0] = math.inf
                                xform_l[1] = math.inf
                            
                            cur_lands[:,pt_idx] = xform_l
            
            if self.do_erase and (random.random() < self.erase_prob):
                #print('  box noise/erase...')

                p_2d_shape = [p.shape[-2], p.shape[-1]]
                box_mean_dim = torch.Tensor([p_2d_shape[0] * 0.15, p_2d_shape[1] * 0.15])
                
                num_boxes = random.randint(1,5)
                
                if self.print_aug_info:
                    print('  Random Corrupt: num. boxes: {}'.format(num_boxes))
                
                for box_idx in range(num_boxes):
                    box_valid = False
                    
                    while not box_valid:
                        # First sample box dims
                        box_dims = torch.round((torch.randn(2) * (box_mean_dim)) + box_mean_dim).long()

                        if (box_dims[0] > 0) and (box_dims[1] > 0) and \
                                (box_dims[0] <= p_2d_shape[0]) and (box_dims[1] <= p_2d_shape[1]):
                            # Next sample box location
                            start_row = random.randint(0, p_2d_shape[0] - box_dims[0])
                            start_col = random.randint(0, p_2d_shape[1] - box_dims[1])

                            box_valid = True
                    
                    p_roi = p[0,start_row:(start_row+box_dims[0]),start_col:(start_col+box_dims[1])]

                    sigma_noise = (p_roi.max() - p_roi.min()) * 0.2
                    
                    p_roi += torch.randn(p_roi.shape) * sigma_noise

        # end data aug

        if need_to_pad_proj:
            p = torch.from_numpy(np.pad(p.numpy(),
                                 ((0, 0), (self.extra_pad, self.extra_pad), (self.extra_pad, self.extra_pad)),
                                 'reflect'))

        if self.do_norm_01_scale:
            p = (p - p.mean()) / p.std()

        h = None
        if self.include_heat_map:
            assert(cur_seg is not None)
            assert(cur_lands is not None)

            num_lands = cur_lands.shape[-1]

            h = torch.zeros(num_lands, 1, cur_seg.shape[-2], cur_seg.shape[-1])

            # "FH-l", "FH-r", "GSN-l", "GSN-r", "IOF-l", "IOF-r", "MOF-l", "MOF-r", "SPS-l", "SPS-r", "IPS-l", "IPS-r"
            #sigma_lut = [ 2.5, 2.5, 7.5, 7.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
            sigma_lut = torch.full([num_lands], 2.5)

            (Y,X) = torch.meshgrid(torch.arange(0, cur_seg.shape[-2]),
                                   torch.arange(0, cur_seg.shape[-1]))
            Y = Y.float()
            X = X.float()

            for land_idx in range(num_lands):
                sigma = sigma_lut[land_idx]

                cur_land = cur_lands[:,land_idx]

                mu_x = cur_land[0]
                mu_y = cur_land[1]

                if not math.isinf(mu_x) and not math.isinf(mu_y):
                    pdf = torch.exp(((X - mu_x).pow(2) + (Y - mu_y).pow(2)) / (sigma * sigma * -2)) / (2 * math.pi * sigma * sigma)
                    #pdf /= pdf.sum() # normalize to sum of 1
                    h[land_idx,0,:,:] = pdf
            #assert(torch.all(torch.isfinite(h)))

        return (p,cur_seg,cur_lands,h)

def get_orig_img_shape(h5_file_path, pat_ind):
    f = h5.File(h5_file_path, 'r')
        
    s = f['{:02d}/projs'.format(pat_ind)].shape
    
    assert(len(s) == 3)
    
    return (s[1], s[2])

def get_num_lands_from_dataset(h5_file_path):
    f = h5.File(h5_file_path, 'r')
    
    num_lands = int(f['land-names/num-lands'].value)

    f.close()

    return num_lands

def get_land_names_from_dataset(h5_file_path):
    f = h5.File(h5_file_path, 'r')
    
    num_lands = int(f['land-names/num-lands'].value)

    land_names = []

    for l in range(num_lands):
        s = f['land-names/land-{:02d}'.format(l)].value
        if (type(s) is bytes) or (type(s) is np.bytes_):
            s = s.decode()
        assert(type(s) is str)

        land_names.append(s)
    
    f.close()

    return land_names


def get_dataset(h5_file_path, pat_inds, num_classes,
                pad_img_dim=0,
                no_seg=False,
                minmax=None,
                data_aug=False,
                train_valid_split=None,
                train_valid_idx=None,
                dup_data_w_left_right_flip=False,
                multi_class_labels=False):
    # classes:
    # 0 --> BG
    # 1 --> Left Hemipelvis
    # 2 --> Right Hemiplevis
    # 3 --> Vertebrae
    # 4 --> Upper Sacrum
    # 5 --> Left Femur
    # 6 --> Right Femur
        
    need_to_scale_data   = False
    need_to_find_min_max = False

    if minmax is not None:
        if (type(minmax) is bool) and minmax:
            need_to_scale_data = True
            print('need to find min/max for preprocessing...')
            need_to_find_min_max = True
            minmax_min =  math.inf
            minmax_max = -math.inf
        elif type(minmax) is tuple:
            minmax_min = minmax[0]
            minmax_max = minmax[1]
            need_to_scale_data = True
            print('using provided min/max for preprocessing: ({}, {})'.format(minmax_min, minmax_max))

    f = h5.File(h5_file_path, 'r')

    all_projs = None
    all_segs  = None
    all_lands = None

    orig_img_shape = None

    for pat_idx in pat_inds:
        pat_g = f['{:02d}'.format(pat_idx)]

        cur_projs_np = pat_g['projs'][:]
        assert(len(cur_projs_np.shape) == 3)

        if orig_img_shape is None:
            orig_img_shape = (cur_projs_np.shape[1], cur_projs_np.shape[2])
        else:
            assert(orig_img_shape[0] == cur_projs_np.shape[1])
            assert(orig_img_shape[1] == cur_projs_np.shape[2])
        
        cur_lands = torch.from_numpy(pat_g['lands'][:])
        assert(cur_lands.shape[0] == cur_projs_np.shape[0])
        assert(torch.all(torch.isfinite(cur_lands)))  # all inputs should be finite

        # mark out of bounds landmarks with inf's
        for img_idx in range(cur_lands.shape[0]):
            for l_idx in range(cur_lands.shape[-1]):
                cur_l = cur_lands[img_idx,:,l_idx]

                if (cur_l[0] < 0) or (cur_l[0] > (orig_img_shape[1]-1)) or \
                   (cur_l[1] < 0) or (cur_l[1] > (orig_img_shape[0]-1)):
                       cur_l[0] = math.inf
                       cur_l[1] = math.inf

        if need_to_find_min_max:
            minmax_min = min(minmax_min, cur_projs_np.min())
            minmax_max = max(minmax_max, cur_projs_np.max())

        cur_projs = torch.from_numpy(cur_projs_np)

        # Need a singleton dimension to represent grayscale data
        cur_projs = cur_projs.view(cur_projs.shape[0], 1, cur_projs.shape[1], cur_projs.shape[2])

        if all_projs is None:
            all_projs = cur_projs
        else:
            all_projs = torch.cat((all_projs, cur_projs))

        if multi_class_labels:
            cur_segs      = None
            cur_segs_dice = torch.from_numpy(pat_g['multi-segs'][:])
            assert(len(cur_segs_dice.shape) == 4)
        else:
            cur_segs = torch.from_numpy(pat_g['segs'][:])
            assert(len(cur_segs.shape) == 3)
            
            cur_segs_dice = torch.zeros(cur_segs.shape[0], num_classes, cur_segs.shape[1], cur_segs.shape[2])
            
            for i in range(cur_segs.shape[0]):
                for c in range(num_classes):
                    cur_segs_dice[i,c,:,:] = cur_segs[i,:,:] == c

        if all_segs is None:
            all_segs = cur_segs_dice.clone().detach()
        else:
            all_segs = torch.cat((all_segs, cur_segs_dice))

        if all_lands is None:
            all_lands = cur_lands.clone().detach()
        else:
            all_lands = torch.cat((all_lands, cur_lands))

        if dup_data_w_left_right_flip:
            all_projs = torch.cat((all_projs, torch.flip(cur_projs, [3])))

            # left/right flip the segmentations
            cur_segs_dice = torch.flip(cur_segs_dice, [3])

            assert(cur_segs_dice.shape[1] == 7)  # TODO: allow for a mapping to be passed
            # update l/r labels
            # 0 BG stays the same
            # 1 left hemipelvis <--> 2 right hemipelvis
            # 3 vertebrae stays the same
            # 4 upper sacrum stays the smae
            # 5 left femur <--> 6 left femur

            def swap_classes(c1, c2):
                tmp_copy  = cur_segs_dice[:,c1,:,:].clone().detach()
                cur_segs_dice[:,c1,:,:] = cur_segs_dice[:,c2,:,:]
                cur_segs_dice[:,c2,:,:] = tmp_copy

            swap_classes(1,2)
            swap_classes(5,6)

            # flip lands and update, etc
            for img_idx in range(cur_lands.shape[0]):
                # do the l/r flip for each landmark
                for l_idx in range(cur_lands.shape[-1]):
                    cur_l = cur_lands[img_idx,:,l_idx]
                    if math.isfinite(cur_l[0]) and math.isfinite(cur_l[1]):
                        cur_l[0] = (orig_img_shape[-1] - 1) - cur_l[0]
                
                # now swap the l/r landmarks
                assert((cur_lands.shape[-1] % 2) == 0)
                for l_idx in range(cur_lands.shape[-1] // 2):
                    tmp_land = cur_lands[img_idx,:,l_idx].clone().detach()
                    cur_lands[img_idx,:,l_idx] = cur_lands[img_idx,:,l_idx+1]
                    cur_lands[img_idx,:,l_idx] = tmp_land
            
            all_segs = torch.cat((all_segs, cur_segs_dice))
            all_lands = torch.cat((all_lands, cur_lands))
    
    # end loop over patients
    
    f.close()
    
    # scale to [0,1] if needed
    if need_to_scale_data:
        assert((minmax_max - minmax_min) > 1.0e-6)
        print('scaling data using min/max: {} , {}'.format(minmax_min, minmax_max))
        all_projs = (all_projs - minmax_min) / (minmax_max - minmax_min)

    def set_helper_vars(ds, do_data_aug):
        ds.prob_of_aug = 0.5 if do_data_aug else 0.0
        
        # stuff in some custom vars
        ds.rob_orig_img_shape = orig_img_shape

        ds.rob_data_is_scaled = need_to_scale_data
        if need_to_scale_data:
            ds.rob_minmax = (minmax_min, minmax_max)

    if (train_valid_split is not None) and (train_valid_split > 0):
        print('split dataset into train/validation')
        assert((0.0 < train_valid_split) and (train_valid_split < 1.0))
        num_train = int(math.ceil(train_valid_split * all_projs.shape[0]))
        num_valid = all_projs.shape[0] - num_train

        all_inds = list(range(all_projs.shape[0]))
        
        if (train_valid_idx is None) or (train_valid_idx[0] is None) or (train_valid_idx[1] is None):
            print('  randomly splitting all complete tensors into training/validation...')
            random.shuffle(all_inds)

            train_inds = all_inds[:num_train]
            valid_inds = all_inds[num_train:]
        else:
            print('  use previously specified split')
            train_inds = train_valid_idx[0]
            valid_inds = train_valid_idx[1]
            assert(len(train_inds) == num_train)
            assert(len(valid_inds) == num_valid)
        
        train_ds = RandomDataAugDataSet(all_projs[train_inds,:,:,:], all_segs[train_inds,:,:,:], all_lands[train_inds,:,:], proj_pad_dim=pad_img_dim)
        set_helper_vars(train_ds, data_aug)
        
        valid_ds = RandomDataAugDataSet(all_projs[valid_inds,:,:,:], all_segs[valid_inds,:,:,:], all_lands[valid_inds,:,:], proj_pad_dim=pad_img_dim)
        set_helper_vars(valid_ds, False)

        return (train_ds, valid_ds, train_inds, valid_inds)
    else:
        ds = RandomDataAugDataSet(all_projs, all_segs, all_lands, proj_pad_dim=pad_img_dim)
        set_helper_vars(ds, data_aug)

        return ds



