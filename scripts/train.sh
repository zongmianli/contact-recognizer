#!/bin/bash


# ------------------------------------------------------------
# Configuration

repo_dir=~/contact-recognizer
exec_path=${repo_dir}/main.py
experiment_dir=${repo_dir}/train/results/debug_train
job_name=debug_train_hands
paths_training_data=${repo_dir}/data/joint_images/handtool_train/hands_120.h5,${repo_dir}/data/joint_images/handtool_train_2/hands_120.h5,${repo_dir}/data/joint_images/parkour_train/hands_120.h5
strides_training_data=5,1,5
models_folder=${repo_dir}/models
joint_name=hands
num_trials=1
num_epochs=10
parameters_id=v1-lr-0.001
resume=${repo_dir}/models/hands_120/checkpoints/v1-lr-0.001_df61e384_hands_120.pth.tar

# ------------------------------------------------------------
exec_settings="${experiment_dir} ${job_name} \
    ${paths_training_data} ${strides_training_data} \
    --models-folder=${models_folder} \
    --joint-name=${joint_name} \
    --num-trials=${num_trials} \
    --num-epochs=${num_epochs} \
    --parameters-id=${parameters_id}"

# if the $resume path is set to a non-empty string
if [ -n "${resume}" ]; then
    exec_settings="${exec_settings} --resume=${resume}"
fi

mkdir -p ${experiment_dir}

python ${exec_path} ${exec_settings}
