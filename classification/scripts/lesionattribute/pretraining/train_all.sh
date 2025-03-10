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

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$PORT ./train/train_classify_pretrained.py \
        --train_image_list config/data/SynLesion/all/image.txt \
        --train_attribute_list config/data/SynLesion/all/json.txt\
        --train_label_list config/data/SynLesion/all/label.txt \
        --valid_image_list config/data/LesionAttribute/all/fold_0/valid_image.txt \
        --valid_label_list config/data/LesionAttribute/all/fold_0/valid_answer.txt \
        --net_type Generic_UNet_classify \
        --input_channel 1 \
        --output_channel 14 \
        --base_feature_number 64 \
        --pretrained_model_path /path/to/stage1_PASTA/checkpoint_latest.pth \
        --model_save_name weights/PASTA_classify/PASTA_classify_pretrained_fold_0 \
        --batch_size 16 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --decay 1e-5 \
        --total_step 20000 \
        --start_step 0 \
        --save_step 1000 \
        --log_freq 100 \
        --accumulation_steps 1 \
        --class_num 5 3 2 2 2 \
        --crop_shape 96 96 96 \

    
