import torch
import shutil
import os
import copy

class Save_checkpoint(object):
    def __init__(self):
        self.pre_check_save_name = ''
        self.pre_best_save_name = ''
        self.delete_pre = False

    def save_checkpoint(self, state, is_best, logger, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
        save_fold = os.path.dirname(filename)
        os.makedirs(save_fold, exist_ok=True)
        if self.delete_pre:
            if os.path.exists(self.pre_check_save_name):
                os.remove(self.pre_check_save_name)
        self.pre_check_save_name = filename
        torch.save(state, filename)
        logger.print('succeffcully save', filename)
        if is_best:
            if self.delete_pre:
                if os.path.exists(self.pre_best_save_name):
                    os.remove(self.pre_best_save_name)
            self.pre_best_save_name = bestname
            shutil.copyfile(filename, bestname)
        self.delete_pre = True

def move_optimizer_to_cpu(optimizer):
    # Move optimizer state to CPU to prevent GPU memory leak
    optimizer_cpu = copy.deepcopy(optimizer)
    for state in optimizer_cpu.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    return optimizer_cpu.state_dict()