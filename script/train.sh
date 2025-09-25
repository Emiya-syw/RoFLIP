set -e
lr=5e-06
bs=64
cd ./src
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HF_ENDPOINT=https://hf-mirror.com
for wdcm in 0.2 0.4 0.6 0.8 1.0; do
for wdtm in 0.2 0.4 0.6 0.8 1.0; do
for wdense in 0.2 0.4 0.6 0.8 1.0; do
output_name=roflip_b_32_${wdcm}_${wdtm}_${wdense}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 -m open_clip_train.main \
            --seed 42 \
            --train-data ./rofa/coco2014_neg.json \
            --dataset-type json \
            --save-frequency 1 \
            --batch-size $bs \
            --lr $lr \
            --beta1 0.9 \
            --beta2 0.98 \
            --eps 1e-06 \
            --wd 0.1 \
            --warmup 50 \
            --epochs 10 \
            --workers 4 \
            --pretrained openai \
            --model ViT-B-32 \
            --cache-dir ./weights \
            --logs-dir Outputs \
            --log-every-n-steps 10 \
            --gather-with-grad \
            --rofclip \
            --wdcm $wdcm \
            --wdtm $wdtm \
            --wdense $wdense \
            --M 5 \
            --margin 0.2 \
            --alpha 0.5 \
            --name $output_name
done
done
done

