import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Prepare dictionary for compatible parameters
    compatible_pretrained_dict = {}
    for key, param in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]) and key in pretrained_dict:
            if param.shape == pretrained_dict[key].shape:
                compatible_pretrained_dict[key] = pretrained_dict[key]
            else:
                print(f"Skipping loading parameter {key} due to shape mismatch: "
                      f"expected {param.shape}, got {pretrained_dict[key].shape}")

    # Update model dictionary with compatible parameters
    model_dict.update(compatible_pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in compatible_pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")

    # Load updated dictionary
    mod.load_state_dict(model_dict)
