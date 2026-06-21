#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
seed=${seed:-2}
gpu=${gpu:-"auto"}
group=${group:-"FedDM"}
tag=${tag:-"1"}

#### dev
ipc=${ipc:-10}
no_aug=${no_aug:-0}
model=${model:-"ConvNet"}
dataset=${dataset:-"CIFAR10"}
init=${init:-"real"}
inner_loop=${inner_loop:--1}
outer_loop=${outer_loop:--1}
num_exp=${num_exp:-1}
## DSA
method=${method:-'DSA'}
opt_net_mom=${opt_net_mom:-0}

batch_real=${batch_real:-256}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


cd ../
python main_feddm.py \
    --dataset $dataset \
    --model $model \
    --ipc $ipc \
    --gpu $gpu --save $id --group $group --tag $tag \
    --no_aug $no_aug \
    --init $init \
    --inner_loop $inner_loop --outer_loop $outer_loop \
    --num_exp $num_exp \
    --match_mode 'whole' \
    --dsa_strategy color_crop_cutout_flip_scale_rotate --method $method \
    --opt_X 'sgd' \
    --opt_net 'sgd' \
    --lr_img 1.0 \
    --opt_net_mom $opt_net_mom \
    --batch_real $batch_real \
    --Iteration 1000 \
    --num_users 10 \
    --alpha 0.5 \
    --extreme 0 \