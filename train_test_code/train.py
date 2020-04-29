# Trains a network.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import shutil
import os.path
import time
import sys

import torch
import torch.nn    as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from unet             import *
from dataset          import *
from util             import *
from dice             import *
from warm_restarts_lr import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # input data file is first positional arg
    parser.add_argument('input_data_file_path', help='Path to the datafile containing projections and segmentations', type=str)

    parser.add_argument('--train-pats', help='comma delimited list of patient IDs used for training', type=str)
    parser.add_argument('--valid-pats', help='comma delimited list of patient IDs used for validation', type=str)
    
    parser.add_argument('--multi-class-seg', help='Use overlapping multiple-class segmentation', action='store_true')
    parser.add_argument('--num-classes', help='The number of label classes to be identified', type=int)
    
    parser.add_argument('--batch-size', help='Number of images each minibatch', type=int, default=1)
    
    parser.add_argument('--unet-img-dim', help='Dimension to adjust input images to before inputting into U-Net', type=int, default=364)

    parser.add_argument('--checkpoint-net', help='Path to network saved as checkpoint', type=str, default='zz_checkpoint.pt')
    parser.add_argument('--best-net', help='Path to network saved with best score on the validation data', type=str, default='zz_best_valid.pt')

    parser.add_argument('--checkpoint-freq', help='Frequency (in terms of epochs) at which to save the network checkpoint to disk.', type=int, default=1)

    parser.add_argument('--no-save-best-valid', help='Do not save best validation netowrk to disk.', action='store_true')

    parser.add_argument('--optim', help='Optimization strategy to use.', type=str, default='sgd')

    parser.add_argument('--lr-sched', help='Learning rate scheduling method. \'cos\' --> Cosine annealing with warm restarts, \'none\' --> fixed LR (at initial), \'plateau\' --> reduce learning rate when validation score plateaus', type=str, default='cos')

    parser.add_argument('--init-lr', help='Initial learning rate for SGN using cosine annealing', type=float, default=1.0e-2)

    parser.add_argument('--lr-patience', help='Patience, in # epochs, when using LR plateau decay', type=int, default=20)
    parser.add_argument('--lr-cooldown', help='Cooldown, in # epochs, when using LR plateau decay', type=int, default=20)

    parser.add_argument('--nesterov', help="Use Nesterov momentum in SGD", action='store_true')

    parser.add_argument('--momentum', help='SGD momentum term', type=float, default=0.9)

    parser.add_argument('--wgt-decay', help='SGD weight decay term', type=float, default=0)

    parser.add_argument('--cos-anneal-epochs', help='Number of epochs in the cosine annealing LR scheduling. When using warm restarts with a growth factor, this is the initial period.', type=int, default=10)
    
    parser.add_argument('--cos-growth', help='Growth factor to use with warm restarts.', type=int, default=2)
    
    parser.add_argument('--save-restart-net', help='Prefix used to save networks before warm restart, file path will be <PREFIX>_XX.pt, where XX is the restart index', type=str)
    
    parser.add_argument('--save-after-n-restarts', help='Save networks prior to warm restart only after this number of restarts have been performed.', type=int, default=0)

    parser.add_argument('--max-num-restarts', help='Maximum number of warm restarts; disabled when <= 0, otherwise overrides --max-num-epochs', type=int, default=-1)

    parser.add_argument('--max-num-epochs', help='Maximum number of epochs', type=int, default=200)
    
    parser.add_argument('--train-loss-txt', help='output file for training loss', type=str, default='train_iter_loss.txt')

    parser.add_argument('--valid-loss-txt', help='output file for validation loss', type=str, default='valid_loss.txt')

    parser.add_argument('--no-gpu', help='Only use CPU - do not use GPU even if it is available', action='store_true')

    parser.add_argument('--max-hours', help='Maximum number of hours to run for; terminates when the program does not expect to be able to complete another epoch. A non-positive value indicates no maximum limit.', type=float, default=-1.0)

    parser.add_argument('--unet-num-lvls', help='Number of levels in the U-Net', type=int, default=5)

    parser.add_argument('--unet-init-feats-exp', help='Number of initial features used in the U-Net, two raised to this power.', type=int, default=4)

    parser.add_argument('--unet-batch-norm', help='Use Batch Normalization in U-Net', action='store_true')

    parser.add_argument('--unet-padding', help='Add padding to preserve image sizes for U-Net', action='store_true')

    parser.add_argument('--unet-no-max-pool', help='Learn downsampling weights instead of max-pooling', action='store_true')

    parser.add_argument('--unet-block-depth', help='Depth of the blocks of convolutions at each level', type=int, default=2)

    parser.add_argument('--data-aug', help='Randomly augment the data', action='store_true')
    parser.add_argument('--use-lands', help='Learn landmark heatmaps', action='store_true')
    parser.add_argument('--heat-coeff', help='Weighting applied to heatmap loss - dice gets one minus this.', type=float, default=0.5)

    parser.add_argument('--dice-valid', help='Use only dice validation loss even when training with dice + heatmap loss', action='store_true')
    parser.add_argument('--unet-no-res', help='Do not use residual connections in U-Net blocks', action='store_true')
    parser.add_argument('--train-valid-split', help='Ratio of training data to keep as training, one minus this is used for validation. Enabled when a value in [0,1] is provided, and overrides the valid-pats flag.', type=float, default=-1.0)

    args = parser.parse_args()

    data_file_path = args.input_data_file_path

    assert(args.train_pats is not None)
    train_pats = [int(i) for i in args.train_pats.split(',')]
    assert(len(train_pats) > 0)

    if args.train_valid_split < 0:
        assert(args.valid_pats is not None)
        valid_pats = [int(i) for i in args.valid_pats.split(',')]
        assert(len(valid_pats) > 0)

    save_best_valid = not args.no_save_best_valid

    num_classes = args.num_classes
    
    do_multi_class = args.multi_class_seg

    batch_size = args.batch_size

    proj_unet_dim = args.unet_img_dim

    checkpoint_filename = args.checkpoint_net
    best_valid_filename = args.best_net

    checkpoint_freq = args.checkpoint_freq

    optim_type = args.optim

    init_lr = args.init_lr

    nesterov = args.nesterov

    momentum = args.momentum
    wgt_decay = args.wgt_decay

    lr_sched_meth = args.lr_sched.lower()

    lr_patience = args.lr_patience
    lr_cooldown = args.lr_cooldown

    lr_sched_num_epochs      = args.cos_anneal_epochs
    lr_restart_growth_factor = args.cos_growth

    max_num_restarts = args.max_num_restarts

    save_restart_net_prefix = args.save_restart_net

    save_after_n_restarts = args.save_after_n_restarts

    num_epochs = args.max_num_epochs
    
    train_loss_txt_path = args.train_loss_txt
    valid_loss_txt_path = args.valid_loss_txt

    max_hours = args.max_hours
    enforce_max_hours = max_hours > 0

    cpu_dev = torch.device('cpu')

    if args.no_gpu:
        dev = cpu_dev
    else:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_valid_split = args.train_valid_split

    unet_num_lvls = args.unet_num_lvls
    unet_init_feats_exp = args.unet_init_feats_exp
    unet_batch_norm = args.unet_batch_norm
    unet_padding = args.unet_padding
    unet_no_max_pool = args.unet_no_max_pool
    unet_use_res = not args.unet_no_res
    unet_block_depth = args.unet_block_depth

    data_aug = args.data_aug

    use_lands = args.use_lands
    num_lands = 0
    if use_lands:
        num_lands = get_num_lands_from_dataset(data_file_path)
        print('num. lands read from file: {}'.format(num_lands))
        assert(num_lands > 0)
    
    heat_coeff = args.heat_coeff

    use_dice_valid = args.dice_valid

    num_restarts = 0

    #data_minmax = True

    train_idx = None
    valid_idx = None

    load_from_checkpoint = os.path.exists(checkpoint_filename)

    prev_state = None
    if load_from_checkpoint:
        print('loading state from checkpoint...')
        prev_state = torch.load(checkpoint_filename)
        
        print('loading unet params from checkpoint state dict...')
        save_best_valid     = prev_state['save-best-valid']
        num_classes         = prev_state['num-classes']
        optim_type          = prev_state['optim-type']
        unet_num_lvls       = prev_state['depth']
        unet_init_feats_exp = prev_state['init-feats-exp']
        unet_batch_norm     = prev_state['batch-norm']
        unet_padding        = prev_state['padding']
        unet_no_max_pool    = prev_state['no-max-pool']
        proj_unet_dim       = prev_state['pad-img-size']
        batch_size          = prev_state['batch-size']
        data_aug            = prev_state['data-aug']
        num_lands           = prev_state['num-lands']
        heat_coeff          = prev_state['heat-coeff']
        use_dice_valid      = prev_state['use-dice-valid']
        unet_use_res        = prev_state['unet-use-res']
        unet_block_depth    = prev_state['unet-block-depth']
        do_multi_class      = prev_state['do-multi-class']

        print('         do multi-class: {}'.format(do_multi_class))
        print('           num. classes: {}'.format(num_classes))
        print('            optim. type: {}'.format(optim_type))
        print('                  depth: {}'.format(unet_num_lvls))
        print('      init. feats. exp.: {}'.format(unet_init_feats_exp))
        print('            batch norm.: {}'.format(unet_batch_norm))
        print('       unet do pad img.: {}'.format(unet_padding))
        print('            no max pool: {}'.format(unet_no_max_pool))
        print('  reflect pad img. dim.: {}'.format(proj_unet_dim))
        print('             batch size: {}'.format(batch_size))
        print('              data aug.: {}'.format(data_aug))
        print('         num. landmarks: {}'.format(num_lands))
        print('    use dice for valid.: {}'.format(use_dice_valid))
        print('          unet use res.: {}'.format(unet_use_res))
        print('       unet block depth: {}'.format(unet_block_depth))

        #if ('minmax-min' in prev_state) and ('minmax-max' in prev_state):
        #    data_minmax = (prev_state['minmax-min'], prev_state['minmax-max'])
        #    print('loaded data min/max from state: {} , {}'.format(*data_minmax))

        nesterov  = prev_state['opt-nesterov']
        momentum  = prev_state['opt-momentum']
        wgt_decay = prev_state['opt-wgt-decay']

        print('        nesterov: {}'.format(nesterov))
        print('        momentum: {}'.format(momentum))
        print('    weight decay: {}'.format(wgt_decay))

        lr_sched_meth            = prev_state['lrs-meth']
        lr_sched_num_epochs      = prev_state['lrs-num-epochs']
        lr_restart_growth_factor = prev_state['lrs-growth-factor']
        max_num_restarts         = prev_state['lrs-max-num-restarts']
        save_restart_net_prefix  = prev_state['lrs-save-restart-net-prefix']
        save_after_n_restarts    = prev_state['lrs-save-after-n-restarts']
        num_restarts             = prev_state['lrs-num-restarts']
        lr_patience              = prev_state['lrs-patience']
        lr_cooldown              = prev_state['lrs-cooldown']

        print('                     LR Sched. Method: {}'.format(lr_sched_meth))
        print('                LR Sched. Num. Epochs: {}'.format(lr_sched_num_epochs))
        print('              LR Sched. Growth Factor: {}'.format(lr_restart_growth_factor))
        print('         LR Sched. Max. Num. Restarts: {}'.format(max_num_restarts))
        print('  LR Sched. Save After Restart Prefix: {}'.format(save_restart_net_prefix))
        print('      LR Sched. Save After N Restarts: {}'.format(save_after_n_restarts))
        print('         LR Sched. Cur. Num. Restarts: {}'.format(num_restarts))
        print('                  LR Plateau Patience: {}'.format(lr_patience))
        print('                  LR Plateau Cooldown: {}'.format(lr_cooldown))

        checkpoint_freq = prev_state['checkpoint-freq']

        print('Checkpoint Freq.: {} epochs'.format(checkpoint_freq))
        
        if train_valid_split >= 0:
            print('loading previous train/valid split inds.')
            train_idx = prev_state['train-idx']
            valid_idx = prev_state['valid-idx']

            assert(train_idx is not None)
            assert(valid_idx is not None)

    enforce_max_num_restarts = max_num_restarts > 0

    lrs_is_cos  = lr_sched_meth == 'cos'
    lrs_none    = lr_sched_meth == 'none'
    lrs_plateau = lr_sched_meth == 'plateau'

    print('initializing training dataset/dataloader')
    train_ds = get_dataset(data_file_path, train_pats,
                           num_classes=num_classes,
                           pad_img_dim=proj_unet_dim,
                           data_aug=data_aug,
                           train_valid_split=train_valid_split,
                           train_valid_idx=(train_idx,valid_idx),
                           dup_data_w_left_right_flip=False,
                           multi_class_labels=do_multi_class)
    if train_valid_split >= 0:
        assert(type(train_ds) is tuple)
        (train_ds, valid_ds, train_idx, valid_idx) = train_ds
    
    #data_minmax = train_ds.rob_minmax
    
    num_data_workers = 8 if data_aug else 0

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_data_workers)

    train_ds_len = len(train_ds)

    print('Length of training dataset: {}'.format(train_ds_len))
   
    if train_valid_split < 0:
        print('initializing validation dataset')
        valid_ds = get_dataset(data_file_path, valid_pats,
                               num_classes=num_classes,
                               pad_img_dim=proj_unet_dim,
                               data_aug=False,
                               multi_class_labels=do_multi_class)
    
    print('Length of validation dataset: {}'.format(len(valid_ds)))

    best_valid_loss = None
    epoch = 0

    print('creating network')
    net = UNet(n_classes=num_classes, depth=unet_num_lvls, wf=unet_init_feats_exp, batch_norm=unet_batch_norm, padding=unet_padding, max_pool=not unet_no_max_pool, num_lands=num_lands, do_res=unet_use_res, block_depth=unet_block_depth)
    
    if load_from_checkpoint:
        net.load_state_dict(prev_state['model-state-dict'])
   
    print('moving network to device...')
    net.to(dev)
    
    print('creating loss function')
    if num_lands > 0:
        print('  Dice + Heatmap Loss...')
        criterion = DiceAndHeatMapLoss2D(skip_bg=False, heatmap_wgt=heat_coeff)
    else:
        print('  Dice only...')
        criterion = DiceLoss2D(skip_bg=False)
    
    lr_sched = None

    if optim_type == 'sgd':
        print('creating SGD optimizer and LR scheduler')
        optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=momentum,
                              weight_decay=wgt_decay, nesterov=nesterov)

        if lrs_is_cos:
            lr_sched = WarmRestartLR(optimizer, init_run_period_epochs=lr_sched_num_epochs, growth_factor=lr_restart_growth_factor)
        elif lrs_plateau:
            lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience, verbose=True, cooldown=lr_cooldown)
        else:
            assert(lrs_none)
        #lr_sched = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    elif optim_type == 'adam':
        print('creating ADAM optimizer')
        optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wgt_decay)
        lr_sched = None
        assert(lrs_none)
    elif optim_type == 'rmsprop':
        print('creating RMSProp optimizer')
        optimizer = optim.RMSprop(net.parameters(), lr=init_lr, weight_decay=wgt_decay, momentum=momentum)
        lr_sched = None
        assert(lrs_none)

    if load_from_checkpoint:
        optimizer.load_state_dict(prev_state['optimizer-state-dict'])
        
        if lr_sched is not None:
            lr_sched.load_state_dict(prev_state['scheduler-state-dict'])
        
        best_valid_loss = prev_state['best-valid-loss']
        epoch = prev_state['epoch']
    
    del prev_state

    train_iter_loss_out = RunningFloatWriter(train_loss_txt_path, new_file=not load_from_checkpoint)
    
    valid_loss_out = RunningFloatWriter(valid_loss_txt_path, new_file=not load_from_checkpoint)

    tot_time_this_session_hours = 0.0
    num_epochs_completed_this_session = 0

    print('Start Training...')
    
    keep_training = True

    while keep_training:
        epoch_start_time = time.time()

        print('Epoch: {:03d}'.format(epoch))

        net.train()

        num_batches = 0
        avg_loss    = 0.0

        running_loss = 0.0
        running_loss_num_iters = int(0.05 * train_ds_len)
        running_loss_iter = 0

        num_examples_run = 0

        for (i, data) in enumerate(train_dl, 0):
            (proj, mask, lands, heat) = data

            projs = proj.to(dev)
            masks = mask.to(dev)

            if num_lands > 0:
                if len(heat.shape) > 4:
                    assert(len(heat.shape) == 5)
                    assert(heat.shape[2] == 1)
                    heat = heat.view(heat.shape[0], heat.shape[1], heat.shape[3], heat.shape[4])
                heats = heat.to(dev)

            optimizer.zero_grad()

            net_out = net(projs)
            if num_lands > 0:
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
            
            loss.backward()

            optimizer.step()
            
            num_examples_run += projs.shape[0]
            if (lr_sched is not None) and lrs_is_cos:
                lr_sched.intra_epoch_step(num_examples_run / train_ds_len)

            l = loss.item()
            
            train_iter_loss_out.write(l)

            avg_loss    += l
            num_batches += 1
            
            running_loss      += l
            running_loss_iter += 1
            if running_loss_iter == running_loss_num_iters:
                print('    Running Avg. Loss: {:.6f}'.format(running_loss / running_loss_num_iters))

                running_loss_iter = 0
                running_loss      = 0.0

        avg_loss /= num_batches

        print('  Running validation')
        (avg_valid_loss, std_valid_loss) = test_dataset(valid_ds, net, dev=dev,
                                                        num_lands=0 if use_dice_valid else num_lands)

        valid_loss_out.write(avg_valid_loss)

        print('  Avg. Training Loss: {:.6f}'.format(avg_loss))
        print('  Validation Loss: {:.6f} +/- {:.6f}'.format(avg_valid_loss, std_valid_loss))
        
        if lr_sched is not None:
            if lrs_plateau:
                lr_sched.step(avg_valid_loss)
            else:
                lr_sched.step()
            
            if lrs_is_cos and lr_sched.just_restarted:
                print('  Next epoch is warm restart...')
                num_restarts += 1

        epoch += 1
       
        new_best_valid = False
        if (best_valid_loss is None) or (avg_valid_loss < best_valid_loss):
            best_valid_loss = avg_valid_loss
            new_best_valid = True
        
        def save_net(net_path):
            tmp_name = '{}.tmp'.format(net_path)
            torch.save({ 'epoch'                : epoch,
                         'model-state-dict'     : net.state_dict(),
                         'optim-type'           : optim_type,
                         'optimizer-state-dict' : optimizer.state_dict(),
                         'scheduler-state-dict' : lr_sched.state_dict() if lr_sched is not None else None,
                         'loss'                 : loss,
                         'best-valid-loss'      : best_valid_loss,
                         'save-best-valid'      : save_best_valid,
                         'num-classes'          : num_classes,
                         'do-multi-class'       : do_multi_class,
                         'depth'                : unet_num_lvls,
                         'init-feats-exp'       : unet_init_feats_exp,
                         'batch-norm'           : unet_batch_norm,
                         'padding'              : unet_padding,
                         'no-max-pool'          : unet_no_max_pool,
                         'pad-img-size'         : proj_unet_dim,
                         'batch-size'           : batch_size,
                         #'minmax-min'           : data_minmax[0],
                         #'minmax-max'           : data_minmax[1],
                         'data-aug'             : data_aug,
                         'opt-nesterov'         : nesterov,
                         'opt-momentum'         : momentum,
                         'opt-wgt-decay'        : wgt_decay,
                         'num-lands'            : num_lands,
                         'heat-coeff'           : heat_coeff,
                         'use-dice-valid'       : use_dice_valid,
                         'unet-use-res'         : unet_use_res,
                         'unet-block-depth'     : unet_block_depth,
                         'lrs-meth'             : lr_sched_meth,
                         'lrs-num-epochs'       : lr_sched_num_epochs,
                         'lrs-growth-factor'    : lr_restart_growth_factor,
                         'lrs-max-num-restarts' : max_num_restarts,
                         'lrs-save-restart-net-prefix' : save_restart_net_prefix,
                         'lrs-save-after-n-restarts' : save_after_n_restarts,
                         'lrs-num-restarts'     : num_restarts,
                         'lrs-patience'         : lr_patience,
                         'lrs-cooldown'         : lr_cooldown,
                         'checkpoint-freq'      : checkpoint_freq,
                         'train-idx'            : train_idx,
                         'valid-idx'            : valid_idx },
                       tmp_name)
            shutil.move(tmp_name, net_path)
        
        net_saved_this_epoch_path = None
        if (epoch % checkpoint_freq) == 0:
            print('  Saving checkpoint')
            save_net(checkpoint_filename)
            net_saved_this_epoch_path = checkpoint_filename

        if new_best_valid and save_best_valid:
            print('  Saving best validation (loss: {:.6f})'.format(best_valid_loss))
            
            # if the checkpoint is saved, just copy the file
            if net_saved_this_epoch_path is not None:
                shutil.copy(net_saved_this_epoch_path, best_valid_filename)
            else:
                save_net(best_valid_filename)
                net_saved_this_epoch_path = best_valid_filename
        
        if lrs_is_cos and lr_sched.just_restarted and (save_restart_net_prefix is not None) and (num_restarts >= save_after_n_restarts):
            restart_net_path = '{}_{:02d}.pt'.format(save_restart_net_prefix, num_restarts - 1)

            print('  Saving network before restart {} to {}'.format(num_restarts, restart_net_path))
            
            if net_saved_this_epoch_path is not None:
                shutil.copy(net_saved_this_epoch_path, restart_net_path)
            else:
                save_net(restart_net_path)
                net_saved_this_epoch_path = restart_net_path

        epoch_end_time = time.time()
        
        this_epoch_hours = (epoch_end_time - epoch_start_time) / (60.0 * 60.0)
        print('  This epoch took {:.4f} hours!'.format(this_epoch_hours))
        
        tot_time_this_session_hours += this_epoch_hours

        num_epochs_completed_this_session += 1
       
        avg_epoch_time_hours = tot_time_this_session_hours / num_epochs_completed_this_session

        print('  Current average epoch runtime: {:.4f} hours'.format(avg_epoch_time_hours))
        
        if enforce_max_hours:
            if (tot_time_this_session_hours + avg_epoch_time_hours) > max_hours:
                print('  Exiting - did not expect to be able to complete next expoch within time limit!')
                keep_training = False
        if enforce_max_num_restarts:
            if num_restarts >= max_num_restarts:
                keep_training = False
                print('  Exiting - maximum number of restarts performed!')
        elif epoch >= num_epochs:
            keep_training = False
            print('  Exiting - maximum number of epochs performed!')

        if not keep_training:
            print('    saving checkpoint before exit!')
            
            if net_saved_this_epoch_path is None:
                save_net(checkpoint_filename)
                net_saved_this_epoch_path = checkpoint_filename
            elif net_saved_this_epoch_path != checkpoint_filename:
                shutil.copy(net_saved_this_epoch_path, checkpoint_filename)

    print('Training Hours: {:.4f}'.format(tot_time_this_session_hours))

