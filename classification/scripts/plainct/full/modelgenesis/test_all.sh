CUDA_VISIBLE_DEVICES=0 python test/test_classify_binary.py \
    --valid_image_list config/data/PlainCT/yourdata/fold_0/valid_image.txt \
    --valid_label_list config/data/PlainCT/yourdata/fold_0/valid_label.txt \
    --net_type UNet3D_modelgenesis_classify \
    --input_channel 1 \
    --output_channel 2 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_0_best_Auc0.6083_3000.tar \
    --class_num 2 \
    --crop_shape 128 128 128 \
    --output_json results/classify/Plain-CT/ModelsGenesis/fold_0.json

CUDA_VISIBLE_DEVICES=0 python test/test_classify_binary.py \
    --valid_image_list config/data/PlainCT/yourdata/fold_1/valid_image.txt \
    --valid_label_list config/data/PlainCT/yourdata/fold_1/valid_label.txt \
    --net_type UNet3D_modelgenesis_classify \
    --input_channel 1 \
    --output_channel 2 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_1_best_Auc0.8750_4000.tar \
    --class_num 2 \
    --crop_shape 128 128 128 \
    --output_json results/classify/Plain-CT/ModelsGenesis/fold_1.json

CUDA_VISIBLE_DEVICES=0 python test/test_classify_binary.py \
    --valid_image_list config/data/PlainCT/yourdata/fold_2/valid_image.txt \
    --valid_label_list config/data/PlainCT/yourdata/fold_2/valid_label.txt \
    --net_type UNet3D_modelgenesis_classify \
    --input_channel 1 \
    --output_channel 2 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_2_best_Auc0.6556_1000.tar \
    --class_num 2 \
    --crop_shape 128 128 128 \
    --output_json results/classify/Plain-CT/ModelsGenesis/fold_2.json

CUDA_VISIBLE_DEVICES=0 python test/test_classify_binary.py \
    --valid_image_list config/data/PlainCT/yourdata/fold_3/valid_image.txt \
    --valid_label_list config/data/PlainCT/yourdata/fold_3/valid_label.txt \
    --net_type UNet3D_modelgenesis_classify \
    --input_channel 1 \
    --output_channel 2 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_3_best_Auc0.8194_3000.tar \
    --class_num 2 \
    --crop_shape 128 128 128 \
    --output_json results/classify/Plain-CT/ModelsGenesis/fold_3.json

CUDA_VISIBLE_DEVICES=0 python test/test_classify_binary.py \
    --valid_image_list config/data/PlainCT/yourdata/fold_4/valid_image.txt \
    --valid_label_list config/data/PlainCT/yourdata/fold_4/valid_label.txt \
    --net_type UNet3D_modelgenesis_classify \
    --input_channel 1 \
    --output_channel 2 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_4_best_Auc0.6444_1000.tar \
    --class_num 2 \
    --crop_shape 128 128 128 \
    --output_json results/classify/Plain-CT/ModelsGenesis/fold_4.json