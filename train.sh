#!/bin/bash

none="_None_"

### usage ###
# . train.sh $f_name $ndim $train_pct $weight_decay $lr $dropout $opt $random_seed

### Main parameters ###
f_name=${1}
ndim=${2-2}
train_pct=${3-80}
weight_decay=${4-0.0}
lr=${5-0.001}
dropout=${6-0.0}
opt=${7-adam}
random_seed=${8-0}

num_samples=10000

func_params="f_name=str(${f_name})"
#func_params="f_name=str(${f_name}),min_x=int(-5),max_x=int(5),min_y=int(-5),max_y=int(5),step_x=int(0.25),step_y=int(0.25)"

noise_params=$none
#noise_params="distribution=str(normal),loc=float(0.0),scale=float(1.0)"

alpha=1.0
beta=1.0
#alpha_beta=$none
alpha_beta="alpha=float(${alpha}),beta=float(${beta})"

## Other parameters
log_dir="../log_files"
max_epochs=10000
lr_scheduler=$none
#lr_scheduler=reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss_y

#ID_params="method=str(twonn)"
ID_params=$none

### wandb ###
use_wandb=False
group_name="${f_name}:ndim=${ndim}-tdf=${train_pct}-wd=${weight_decay}-lr=${lr}-d=${dropout}-opt=${opt}"
if [ $alpha_beta != $none ]; then
    group_name="${group_name}-alpha=${alpha}-beta=${beta}"
fi
wandb_entity="grokking_ppsp"
wandb_project="learning_with_derivative"

exp_id="${group_name}"

### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` ###
#early_stopping_grokking=$none
early_stopping_grokking="patience=int(1000),metric=str(val_loss_y),metric_threshold=float(0.0)"

python train.py \
	--exp_id $exp_id \
	--log_dir "${log_dir}/${random_seed}" \
	--func_params $func_params \
	--noise_params $noise_params \
	--ndim $ndim \
	--num_samples $num_samples \
	--normalize False \
	--hidden_dim 512 \
	--n_layers 1 \
	--dropout $dropout \
	--train_pct $train_pct \
	--batch_size 512 \
	--optimizer "${opt},lr=${lr},weight_decay=${weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001" \
	--alpha_beta $alpha_beta \
	--lr_scheduler $lr_scheduler \
	--max_epochs $max_epochs \
	--validation_metrics val_loss_y \
	--checkpoint_path $none \
	--every_n_epochs 100 \
	--every_n_epochs_show 200 \
	--save_top_k -1 \
	--use_wandb $use_wandb \
	--wandb_entity $wandb_entity \
	--wandb_project $wandb_project \
	--group_name $group_name \
	--group_vars $none \
	--ID_params $ID_params \
	--accelerator auto \
	--devices auto \
	--random_seed $random_seed \
	--early_stopping_grokking $early_stopping_grokking \
#	--early_stopping_patience 1000000000 \
#	--model_name epoch=88-val_loss_y=13.6392.ckpt \

#filename=train.sh 
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 