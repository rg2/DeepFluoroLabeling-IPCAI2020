# Performs "testing" on the network. Estimates the segmentations and
# landmark heatmaps for each projection of a specific specimen.
# Can use an ensemble of networks to obtain the estimates (via averaging).
#
# Copyright (C) 2019-2023 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse

import torch

import h5py as h5

from unet    import *
from dataset import *
from util    import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ensemble segmentation and heatmap estimation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # input data file is first positional arg
    parser.add_argument('input_data_file_path', help='Path to the datafile containing projections', type=str)
    
    # output data file is second positional arg
    parser.add_argument('output_data_file_path', help='Path to the output datafile containing segmentations', type=str)
    
    # network files used for ensemble
    parser.add_argument('--nets', help='Paths to the networks used to perform segmentation - specify this after the positional arguments', type=str, nargs='+')
    
    parser.add_argument('--pats', help='comma delimited list of patient IDs used for testing', type=str)
    
    parser.add_argument('--no-gpu', help='Only use CPU - do not use GPU even if it is available', action='store_true')
    
    parser.add_argument('--times', help='Path to file storing runtimes for each image', type=str, default='')

    args = parser.parse_args()

    network_paths = args.nets

    src_data_file_path = args.input_data_file_path
    dst_data_file_path = args.output_data_file_path
    
    assert(args.pats is not None)
    test_pats = [int(i) for i in args.pats.split(',')]
    assert(len(test_pats) > 0)
    
    dev = get_device(no_gpu=args.no_gpu)
    
    torch_map_loc = None

    if args.no_gpu:
        torch_map_loc = 'cpu'

    nets = []
    for net_path in network_paths:
        print('  loading state from disk for: {}'.format(net_path))
        
        state = torch.load(net_path, map_location=torch_map_loc)
        
        print('  loading unet params from checkpoint state dict...')
        num_classes         = state['num-classes']
        unet_num_lvls       = state['depth']
        unet_init_feats_exp = state['init-feats-exp']
        unet_batch_norm     = state['batch-norm']
        unet_padding        = state['padding']
        unet_no_max_pool    = state['no-max-pool']
        unet_use_res        = state['unet-use-res']
        unet_block_depth    = state['unet-block-depth']
        proj_unet_dim       = state['pad-img-size']
        batch_size          = state['batch-size']
        num_lands           = state['num-lands']
        epoch               = state['epoch']
        loss                = state['loss']
        best_valid_loss     = state['best-valid-loss']

        print('             num. classes: {}'.format(num_classes))
        print('                    depth: {}'.format(unet_num_lvls))
        print('        init. feats. exp.: {}'.format(unet_init_feats_exp))
        print('              batch norm.: {}'.format(unet_batch_norm))
        print('         unet do pad img.: {}'.format(unet_padding))
        print('              no max pool: {}'.format(unet_no_max_pool))
        print('    reflect pad img. dim.: {}'.format(proj_unet_dim))
        print('            unet use res.: {}'.format(unet_use_res))
        print('         unet block depth: {}'.format(unet_block_depth))
        print('               batch size: {}'.format(batch_size))
        print('              num. lands.: {}'.format(num_lands))
        
        print('          Last Epoch: {}'.format(epoch))
        print('           Last Loss: {}'.format(loss.item()))
        print('    Best Valid. Loss: {}'.format(best_valid_loss))


        print('    creating network')
        net = UNet(n_classes=num_classes, depth=unet_num_lvls, wf=unet_init_feats_exp, batch_norm=unet_batch_norm, padding=unet_padding, max_pool=not unet_no_max_pool,
                   num_lands=num_lands, do_res=unet_use_res, block_depth=unet_block_depth)
    
        net.load_state_dict(state['model-state-dict'])

        del state

        print('  moving network to device...')
        net.to(dev)
        
        nets.append(net)
  
    land_names = None
    if num_lands > 0:
        land_names = get_land_names_from_dataset(src_data_file_path)
        assert(len(land_names) == num_lands)

    print('initializing testing dataset')
    test_ds = get_dataset(src_data_file_path, test_pats, num_classes=num_classes,
                          pad_img_dim=proj_unet_dim, no_seg=True)
    
    print('Length of testing dataset: {}'.format(len(test_ds)))

    print('opening destination file for writing')
    f = h5.File(dst_data_file_path, 'w')

    # save off the landmark names
    if land_names:
        land_names_g = f.create_group('land-names')
        land_names_g['num-lands'] = num_lands

        for l in range(num_lands):
            land_names_g['land-{:02d}'.format(l)] = land_names[l]

    times = []

    print('running network on projections')
    seg_dataset_ensemble(test_ds, nets, f, dev=dev, num_lands=num_lands, times=times)

    print('closing file...')
    f.flush()
    f.close()
    
    if args.times:
        times_out = open(args.times, 'w')
        for t in times:
            times_out.write('{:.6f}\n'.format(t))
        times_out.flush()
        times_out.close()



