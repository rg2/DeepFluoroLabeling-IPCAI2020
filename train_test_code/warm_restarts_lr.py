# Implementation of SGDR: Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
# The PyTorch torch.optim.lr_scheduler.CosineAnnealingLR class (as of version 1.2) does not adjust the 
# period of the cosine annealing schedule (warm restart), or allow intra-epoch LR updates.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import math

import torch.optim.lr_scheduler as lrs

class WarmRestartLR(lrs._LRScheduler):
    def __init__(self, optimizer, init_run_period_epochs=10, lr_min=0, last_epoch=-1, growth_factor=2):
        self.cur_run_period_epochs = init_run_period_epochs
        
        self.lr_min = lr_min
        
        self.next_restart_epoch = init_run_period_epochs
        
        self.last_restart_epoch = last_epoch if last_epoch >= 0 else 0

        self.period_growth_factor = growth_factor
        
        self.cur_epoch_ratio = 0
        
        self.just_restarted = False

        super().__init__(optimizer, last_epoch)
    
    def intra_epoch_step(self, epoch_ratio):
        self.cur_epoch_ratio = epoch_ratio
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, epoch=None):
        self.cur_epoch_ratio = 0

        super().step(epoch)

        if self.last_epoch >= self.next_restart_epoch:
            print('WARM RESTART AFTER PERIOD OF {} EPOCHS'.format(self.cur_run_period_epochs))

            self.last_restart_epoch = self.next_restart_epoch

            self.cur_run_period_epochs *= self.period_growth_factor

            self.next_restart_epoch += self.cur_run_period_epochs

            self.just_restarted = True
        else:
            self.just_restarted = False

    def get_lr(self):
        assert((-1.0e-12 < self.cur_epoch_ratio) and (self.cur_epoch_ratio < (1 + 1.0e-12)))

        shift_cos = 1 + math.cos(math.pi * (self.last_epoch - self.last_restart_epoch + self.cur_epoch_ratio) / self.cur_run_period_epochs)

        new_lrs = [self.lr_min + (((base_lr - self.lr_min) / 2) * shift_cos) for base_lr in self.base_lrs]
        #print(new_lrs)
        return new_lrs
