export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT ./train/train_classify.py \
        --train_image_list config/data/LesionAttribute/all/fold_0/train_image.txt \
        --train_label_list config/data/LesionAttribute/all/fold_0/train_answer.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_0/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_0/valid_answer.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_fold_0 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 10000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT ./train/train_classify.py \
        --train_image_list config/data/LesionAttribute/all/fold_1/train_image.txt \
        --train_label_list config/data/LesionAttribute/all/fold_1/train_answer.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_1/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_1/valid_answer.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_fold_1 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 10000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT ./train/train_classify.py \
        --train_image_list config/data/LesionAttribute/all/fold_2/train_image.txt \
        --train_label_list config/data/LesionAttribute/all/fold_2/train_answer.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_2/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_2/valid_answer.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_fold_2 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 10000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 

    
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT ./train/train_classify.py \
        --train_image_list config/data/LesionAttribute/all/fold_3/train_image.txt \
        --train_label_list config/data/LesionAttribute/all/fold_3/train_answer.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_3/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_3/valid_answer.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_fold_3 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 10000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT ./train/train_classify.py \
        --train_image_list config/data/LesionAttribute/all/fold_4/train_image.txt \
        --train_label_list config/data/LesionAttribute/all/fold_4/train_answer.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_4/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_4/valid_answer.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_fold_4 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 10000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 

    