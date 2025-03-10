from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
import json


def main(args):
    indir = args.indir
    outdir = args.outdir
    split_path = args.split_json
    trainer = args.trainer
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    dataset_name = split_path.split('/')[-2]
    for fold_index in range(5):
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, f'{dataset_name}/{trainer}__nnUNetPlans__3d_fullres'),
            use_folds=(fold_index,),
            checkpoint_name='checkpoint_final.pth',
        )
        
        os.makedirs(outdir, exist_ok=True)
        with open(split_path, 'r') as f:
            split_ls = json.load(f)

        split_valid_ls = split_ls[fold_index]['val']
        input_valid_ls = [[join(indir, valid_name+'_0000.nii.gz', )] for valid_name in split_valid_ls]
        output_valid_ls = [join(outdir, valid_name+'.nii.gz', ) for valid_name in split_valid_ls]
        
        predictor.predict_from_files(input_valid_ls,
                                    output_valid_ls,
                                    save_probabilities=False, overwrite=True,
                                    num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-indir', type=str, required=True, help='/path/to/your/nnUNet_raw/Dataset00x_xxx/imagesTr')
    parser.add_argument('-outdir', type=str, required=True, help='/path/to/your/nnUNet_raw/Dataset00x_xxx/PASTA_ft_pred')
    parser.add_argument('-split_json', type=str, required=True, help='/path/to/your/nnUNet_raw/Dataset00x_xxx/splits_final.json')
    parser.add_argument('-trainer', type=str, default='PASTATrainer_ft', help='nnUNet trainer name')
    args = parser.parse_args()
    
    main(args)
