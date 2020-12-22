#!/bin/bash

train_file=diventura.py
dataset_dir="path to data"
log_dir="path to network folder"

cd ${log_dir}
source ~/.bashrc

source ~/venv/bin/activate

cd ~/maskrcnn_merge/samples/diventura/

python3 ${train_file} detect --dataset ${dataset_dir} --subset tmps --logs ${log_dir} --weights ${log_dir}/mask_rcnn_diventura_15000.h5 --loss aleatoric
