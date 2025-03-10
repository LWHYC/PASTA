CUDA_VISIBLE_DEVICES=2 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_0/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_0/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/Generic_UNet_classify/Generic_UNet_classify_fold_0_best_Acc0.7994_7000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/UNet/fold_0.json

CUDA_VISIBLE_DEVICES=2 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_1/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_1/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/Generic_UNet_classify/Generic_UNet_classify_fold_1_best_Acc0.7838_8000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/UNet/fold_1.json

CUDA_VISIBLE_DEVICES=2 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_2/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_2/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/Generic_UNet_classify/Generic_UNet_classify_fold_2_best_Acc0.8156_6000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/UNet/fold_2.json

CUDA_VISIBLE_DEVICES=2 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_3/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_3/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/Generic_UNet_classify/Generic_UNet_classify_fold_3_best_Acc0.8039_9000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/UNet/fold_3.json

CUDA_VISIBLE_DEVICES=2 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_4/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_4/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/Generic_UNet_classify/Generic_UNet_classify_fold_4_best_Acc0.8020_5000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/UNet/fold_4.json