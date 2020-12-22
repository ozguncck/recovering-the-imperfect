#!/bin/bash

name="" 

train_file=diventura.py
dataset_dir="path to data"
log_dir="path to network folder"


cd ${log_dir}
source ~/.bashrc

source ~/venv/bin/activate


snapshotIter="$(ls mask_rcnn_diventura_*.h5 | grep -o "_[0-9]*[.]" | sed -e "s/^_//" -e "s/.$//" | xargs printf "%05d\n" | sort -n | tail -1)"
snap="$(ls mask_rcnn_diventura_*.h5 | grep -o "_[0-9]*[.]" | sed -e "s/^_//" -e "s/.$//" | sort -n | tail -1)"

echo $snapshotIter
echo $snap

cd ~/maskrcnn/samples/diventura/

if [ "x${snap}" != "x" ]; then
    python3 ${train_file} train --dataset=${dataset_dir} --subset train_final --logs=${log_dir} --weights=${log_dir}/mask_rcnn_diventura_${snap}.h5 --loss=hyp_aleatoric

else
    python3 ${train_file} train --dataset=${dataset_dir} --subset train_final --logs=${log_dir} --loss=hyp_aleatoric
fi


 
