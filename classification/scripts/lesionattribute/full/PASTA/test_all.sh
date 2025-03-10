CUDA_VISIBLE_DEVICES=0 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_0/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_0/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/PASTA_classify/Generic_UNet_classify_fold_0_best_Acc0.8510_5000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/PASTA/fold_0.json

CUDA_VISIBLE_DEVICES=0 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_1/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_1/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/PASTA_classify/Generic_UNet_classify_fold_1_best_Acc0.8201_3000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/PASTA/fold_1.json

CUDA_VISIBLE_DEVICES=0 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_2/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_2/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/PASTA_classify/Generic_UNet_classify_fold_2_best_Acc0.8456_9000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/PASTA/fold_2.json

CUDA_VISIBLE_DEVICES=0 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_3/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_3/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/PASTA_classify/Generic_UNet_classify_fold_3_best_Acc0.8302_9000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/PASTA/fold_3.json

CUDA_VISIBLE_DEVICES=0 python ./test/test_classify.py \
    --valid_image_list config/data/LesionAttribute/all/fold_4/valid_image.txt \
    --valid_label_list config/data/LesionAttribute/all/fold_4/valid_answer.txt \
    --net_type Generic_UNet_classify \
    --input_channel 1 \
    --output_channel 14 \
    --base_feature_number 64 \
    --batch_size 1 \
    --pretrained_model_path weights/PASTA_classify/Generic_UNet_classify_fold_4_best_Acc0.8365_10000.tar \
    --class_num 5 3 2 2 2 \
    --crop_shape 96 96 96 \
    --output_json results/classify/PASTA/fold_4.json