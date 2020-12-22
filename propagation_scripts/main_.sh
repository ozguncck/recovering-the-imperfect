#!/bin/bash

source /path/to/virtualenv
search_dir=/path/to/data/
train_file=/path/to/inference/file.py
weight_file=/path/to/weight/file.h5

for entry in ${search_dir}*.tif
do
  echo ${entry}
  entry_no_space="$(echo -e "${entry}" | tr -d '[:space:]')"
  if [ "${entry}" != "${entry_no_space}" ]; then
    mv "${entry}" "${entry_no_space}"
  fi
  subset=$(basename "${entry_no_space}" .tif)  
  python3 gen-h5.py --path ${entry_no_space}
  width=$(identify -format "%w" "${search_dir}/${subset}/ch0.tif[0]")> /dev/null
  height=$(identify -format "%h" "${search_dir}/${subset}/ch0.tif[0]")> /dev/null
  python3 ${train_file} detect --dataset ${search_dir} --subset ${subset} --logs ${search_dir} --weights ${weight_file} --loss aleatoric
  matlab -nodisplay -nosplash -nodesktop -r "IoU_track_uncertainty('${search_dir}results_${subset}_15000/',${height},${width});exit"
  python3 final_tracking.py --path ${search_dir}results_${subset}_15000/
done




