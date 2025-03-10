CUDA_VISIBLE_DEVICES=1 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_0/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_0/valid_answer.txt \
    --net_type UNet3D_suprem_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_suprem_classify/UNet3D_suprem_classify_fold_0_best_Acc0.7781_9000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/SupreM/fold_0.json

CUDA_VISIBLE_DEVICES=1 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_1/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_1/valid_answer.txt \
    --net_type UNet3D_suprem_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_suprem_classify/UNet3D_suprem_classify_fold_1_best_Acc0.7805_8000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/SupreM/fold_1.json

CUDA_VISIBLE_DEVICES=1 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_2/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_2/valid_answer.txt \
    --net_type UNet3D_suprem_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_suprem_classify/UNet3D_suprem_classify_fold_2_best_Acc0.7687_10000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/SupreM/fold_2.json

CUDA_VISIBLE_DEVICES=1 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_3/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_3/valid_answer.txt \
    --net_type UNet3D_suprem_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_suprem_classify/UNet3D_suprem_classify_fold_3_best_Acc0.7731_5000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/SupreM/fold_3.json

CUDA_VISIBLE_DEVICES=1 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_4/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_4/valid_answer.txt \
    --net_type UNet3D_suprem_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/UNet3D_suprem_classify/UNet3D_suprem_classify_fold_4_best_Acc0.7754_4000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/SupreM/fold_4.json