export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_binary.py \
        --train_image_list config/data/PlainCT/yourdata/fold_0/train_image.txt \
        --train_label_list config/data/PlainCT/yourdata/fold_0/train_label.txt \
        --valid_image_list config/data/PlainCT/yourdata/fold_0/valid_image.txt \
        --valid_label_list config/data/PlainCT/yourdata/fold_0/valid_label.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 2 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_0 \
        --batch_size 8 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 5000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 2 \
        --class_weight 1 10 \
        --crop_shape 128 128 128 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_binary.py \
        --train_image_list config/data/PlainCT/yourdata/fold_1/train_image.txt \
        --train_label_list config/data/PlainCT/yourdata/fold_1/train_label.txt \
        --valid_image_list config/data/PlainCT/yourdata/fold_1/valid_image.txt \
        --valid_label_list config/data/PlainCT/yourdata/fold_1/valid_label.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 2 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_1 \
        --batch_size 8 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 5000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 2 \
        --class_weight 1 10 \
        --crop_shape 128 128 128 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_binary.py \
        --train_image_list config/data/PlainCT/yourdata/fold_2/train_image.txt \
        --train_label_list config/data/PlainCT/yourdata/fold_2/train_label.txt \
        --valid_image_list config/data/PlainCT/yourdata/fold_2/valid_image.txt \
        --valid_label_list config/data/PlainCT/yourdata/fold_2/valid_label.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 2 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_2 \
        --batch_size 8 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 5000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 2 \
        --class_weight 1 10 \
        --crop_shape 128 128 128 

    
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_binary.py \
        --train_image_list config/data/PlainCT/yourdata/fold_3/train_image.txt \
        --train_label_list config/data/PlainCT/yourdata/fold_3/train_label.txt \
        --valid_image_list config/data/PlainCT/yourdata/fold_3/valid_image.txt \
        --valid_label_list config/data/PlainCT/yourdata/fold_3/valid_label.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 2 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_3 \
        --batch_size 8 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 5000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 2 \
        --class_weight 1 10 \
        --crop_shape 128 128 128 

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=2000
export NCCL_NET_GDR_LEVEL=5

set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_binary.py \
        --train_image_list config/data/PlainCT/yourdata/fold_4/train_image.txt \
        --train_label_list config/data/PlainCT/yourdata/fold_4/train_label.txt \
        --valid_image_list config/data/PlainCT/yourdata/fold_4/valid_image.txt \
        --valid_label_list config/data/PlainCT/yourdata/fold_4/valid_label.txt \
        --net_type UNet3D_modelgenesis_classify \
        --input_channel 1 \
        --output_channel 2 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/modelgenesis/new_Genesis_Chest_CT.pth \
        --model_save_name weights/UNet3D_modelgenesis_classify/UNet3D_modelgenesis_classify_plainct_fold_4 \
        --batch_size 8 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 5000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 2 \
        --class_weight 1 10 \
        --crop_shape 128 128 128 
