#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
project=${1}
cfg=${2}
model=${3}
file_num=`find ${project} -name exp*.log -type f |wc -l`
let file_num++
echo check log at ${project}/exp${file_num}.log
echo config=${cfg} project=${project} model=${model}
nohup python train.py --cfg  ${cfg} \
                      --model ${model} \
                      --project ${project} > ${project}/exp${file_num}.log 2>&1 &
# tensorboard --logdir ${log_dir} --port=6006 --host=0.0.0.0
