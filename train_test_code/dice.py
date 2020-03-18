# Differentiable dice loss
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch

import torch.nn.modules.loss

from ncc import ncc_2d

class DiceLoss2D(torch.nn.modules.loss._Loss):
    def __init__(self, skip_bg=True):
        super(DiceLoss2D, self).__init__()
        
        self.skip_bg = skip_bg

    def forward(self, input, target):
        # Add this to numerator and denominator to avoid divide by zero when nothing is segmented
        # and ground truth is also empty (denominator term).
        # Also allow a Dice of 1 (-1) for this case (both terms).
        eps = 1.0e-4
        
        if self.skip_bg:
            # numerator of Dice, for each class except class 0 (background)
            # multiply by -2 (usually +2), since we are minimizing the objective function and we want to maximize Dice
            numerators = -2 * torch.sum(torch.sum(target[:,1:,:,:] * input[:,1:,:,:], dim=3), dim=2) + eps

            # denominator of Dice, for each class except class 0 (background)
            denominators = torch.sum(torch.sum(target[:,1:,:,:] * target[:,1:,:,:], dim=3), dim=2) + \
                             torch.sum(torch.sum(input[:,1:,:,:] * input[:,1:,:,:], dim=3), dim=2) + eps

            # minus one to exclude the background class
            num_classes = input.shape[1] - 1
        else:
            # numerator of Dice, for each class
            # multiply by -2 (usually +2), since we are minimizing the objective function and we want to maximize Dice
            numerators = -2 * torch.sum(torch.sum(target[:,:,:,:] * input[:,:,:,:], dim=3), dim=2) + eps

            # denominator of Dice, for each class
            denominators = torch.sum(torch.sum(target[:,:,:,:] * target[:,:,:,:], dim=3), dim=2) + \
                             torch.sum(torch.sum(input[:,:,:,:] * input[:,:,:,:], dim=3), dim=2) + eps
            
            num_classes = input.shape[1]

        # Dice coefficients for each image in the batch, for each class
        dices = numerators / denominators

        # compute average Dice score for each image in the batch
        avg_dices = torch.sum(dices, dim=1) / num_classes
        
        # compute average over the batch
        return torch.mean(avg_dices)

class DiceAndHeatMapLoss2D(torch.nn.modules.loss._Loss):
    def __init__(self, skip_bg=True, heatmap_wgt=0.5):
        super(DiceAndHeatMapLoss2D, self).__init__()
        
        self.dice_loss = DiceLoss2D(skip_bg=skip_bg)
       
        assert((heatmap_wgt > 1.0e-8) and (heatmap_wgt < (1 + 1.0e-8)))
        self.heatmap_wgt = heatmap_wgt
        self.dice_wgt    = 1 - heatmap_wgt

    def forward(self, input, target):
        in_seg      = input[0]
        in_heatmaps = input[1]

        tgt_seg      = target[0]
        tgt_heatmaps = target[1]

        num_lands = tgt_heatmaps.shape[1]

        # L2 Loss
        #hm_errs = (in_heatmaps - tgt_heatmaps).pow(2)
        #avg_hm_errs = torch.sum(torch.sum(torch.sum(hm_errs, dim=3), dim=2), dim=1) / num_lands
        #return self.dice_loss(in_seg, tgt_seg) + (self.heatmap_wgt * torch.mean(avg_hm_errs))
        
        ncc_losses = ncc_2d(in_heatmaps, tgt_heatmaps)
        
        # negation since we are minmizing, normalize output in range [-1,0]
        ncc_losses = (ncc_losses + 1) * -0.5
        
        return (self.dice_wgt * self.dice_loss(in_seg, tgt_seg)) + (self.heatmap_wgt * torch.mean(ncc_losses))
        


