from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupreMeTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-2
        print('!!!!!!!!!!!SupreMeTrainer!!!!!!!!!!!')
    @staticmethod
    def build_network_architecture(network_arch_class_name,
                                   network_arch_init_kwargs,
                                   network_arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_segmentation_heads,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        num_classes = num_segmentation_heads
        return UNet3D(num_input_channels, num_classes, 'relu', enable_deep_supervision=enable_deep_supervision)

class SupreMeTrainer_ft(SupreMeTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 500
        
class SupreMeTrainer_fewshot(SupreMeTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10
        self.initial_lr = 1e-3
        self.num_iterations_per_epoch = 200
        self.num_valid_iterations_per_epoch = 50

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)

        # 判断模型当前是否处于训练模式
        training = self.training  # self.training 是由 PyTorch 自动管理的属性
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('activation not correct!')

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.sigmoid(self.final_conv(x))
        out = self.final_conv(x)
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, input_channels, n_class=1, act='relu', enable_deep_supervision=False):
        super(UNet3D, self).__init__()

        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        
        self.do_ds = enable_deep_supervision
        
        self.down_tr64 = DownTransition(input_channels,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, n_class)
        
        self.seg_outputs_0 = nn.Conv3d(512, n_class, 1, 1, 0, 1, 1, False)
        self.seg_outputs_1 = nn.Conv3d(256, n_class, 1, 1, 0, 1, 1, False)
        self.seg_outputs_2 = nn.Conv3d(128, n_class, 1, 1, 0, 1, 1, False)

    def forward(self, x):
        seg_outputs = []
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512 = self.down_tr512(self.out256)
        if self.decoder.deep_supervision:
            seg_outputs.append(self.seg_outputs_0(self.out512[0]))
        self.out_up_256 = self.up_tr256(self.out512[0],self.skip_out256)
        if self.decoder.deep_supervision:
            seg_outputs.append(self.seg_outputs_1(self.out_up_256))
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        if self.decoder.deep_supervision:
            seg_outputs.append(self.seg_outputs_2(self.out_up_128))
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        seg_outputs.append(self.out_tr(self.out_up_64))

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.do_ds:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r