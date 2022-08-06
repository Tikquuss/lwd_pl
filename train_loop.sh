#!/bin/bash

# Usage : ./train_loop.sh $f_name $ndim

#f_name=Styblinski-Tang
f_name=${1-"Styblinski-Tang"}
ndim=${2-2}

for train_pct in 80; do {
for weight_decay in 0.0; do {
for lr in 0.001; do {
for dropout in 0.0; do {
for opt in adam; do {
for random_seed in 0 100; do {
. train.sh $f_name $ndim $train_pct $weight_decay $lr $dropout $opt $random_seed
} done
} done
} done
} done
} done
} done
