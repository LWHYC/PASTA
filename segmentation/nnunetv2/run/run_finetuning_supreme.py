from unittest.mock import patch
from run_training import run_training_entry
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

def load_PASTA_pretrained_weights(network, fname, verbose=False):

    saved_model = torch.load(fname)
    if fname.endswith('pth'):
        pretrained_dict = saved_model['net']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']
    
    new_pretrained_dict = {key.replace('module.backbone.', ''): pretrained_dict[key] for key in pretrained_dict.keys()}
    
    skip_strings_in_pretrained = [
        'seg_outputs',
        'tu.0',
        'organ_embedding',
        'final_conv'
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # Adjust for multimodal inputs
    num_inputs = model_dict['down_tr64.ops.0.conv1.weight'].shape[1]
    if num_inputs > 1:
        pretrained_conv1_weight = new_pretrained_dict['down_tr64.ops.0.conv1.weight']
        new_pretrained_dict['down_tr64.ops.0.conv1.weight'] = pretrained_conv1_weight.repeat(1, num_inputs, 1, 1, 1)

    print('model_dick key', model_dict.keys())
    print('new_pretrained_dict key', new_pretrained_dict.keys())
    
    # Verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in new_pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == new_pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{new_pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model " \
                f"does not seem to be compatible with your network."

    new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(new_pretrained_dict)
    
    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in new_pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict, strict=False)

if __name__ == '__main__':
    with patch("run_training.load_pretrained_weights", load_PASTA_pretrained_weights):
        run_training_entry()
