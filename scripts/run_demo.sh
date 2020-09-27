#!/bin/bash


# ------------------------------------------------------------
# Configuration

repo_dir=~/contact-recognizer
save_name=demo
image_folder=my_imagefolder

info_path=${repo_dir}/data/full_images/${image_folder}/data_info.pkl

mkdir ${repo_dir}/results
save_path=${repo_dir}/results/${save_name}.pkl


# ------------------------------------------------------------
echo "(run_demo.sh) Testing pre-trained models on the demo data ..."

joint_name=neck
patch_size=180
checkpoint=v1-lr-0.005_7172a45d_neck_180.pth.tar
data_path=${repo_dir}/data/joint_images/${image_folder}/${joint_name}_${patch_size}.h5
resume=${repo_dir}/models/${joint_name}_${patch_size}/checkpoints/${checkpoint}
echo $resume
python test_model.py ${joint_name} ${resume} ${data_path} ${info_path} --save-path=${save_path}

joint_name=hands
patch_size=120
checkpoint=v1-lr-0.001_df61e384_hands_120.pth.tar
data_path=${repo_dir}/data/joint_images/${image_folder}/${joint_name}_${patch_size}.h5
resume=${repo_dir}/models/${joint_name}_${patch_size}/checkpoints/${checkpoint}
echo $resume
python test_model.py ${joint_name} ${resume} ${data_path} ${info_path} --save-path=${save_path}

joint_name=knees
patch_size=120
checkpoint=v1-lr-0.001_127696d5_knees_120.pth.tar
data_path=${repo_dir}/data/joint_images/${image_folder}/${joint_name}_${patch_size}.h5
resume=${repo_dir}/models/${joint_name}_${patch_size}/checkpoints/${checkpoint}
echo $resume
python test_model.py ${joint_name} ${resume} ${data_path} ${info_path} --save-path=${save_path}

joint_name=soles
patch_size=120
checkpoint=v1-lr-0.0012_0ba883bc_soles_120.pth.tar
data_path=${repo_dir}/data/joint_images/${image_folder}/${joint_name}_${patch_size}.h5
resume=${repo_dir}/models/${joint_name}_${patch_size}/checkpoints/${checkpoint}
echo $resume
python test_model.py ${joint_name} ${resume} ${data_path} ${info_path} --save-path=${save_path}

joint_name=toes
patch_size=120
checkpoint=v1-lr-0.0012_db801a67_toes_120.pth.tar
data_path=${repo_dir}/data/joint_images/${image_folder}/${joint_name}_${patch_size}.h5
resume=${repo_dir}/models/${joint_name}_${patch_size}/checkpoints/${checkpoint}
echo $resume
python test_model.py ${joint_name} ${resume} ${data_path} ${info_path} --save-path=${save_path}


# ------------------------------------------------------------
echo "(run_demo.sh) Visualizing predicted contact states ..."

img_dir=${repo_dir}/data/full_images/${image_folder}
vis_dir=${repo_dir}/results/${save_name}_vis
joint_names=neck,hands,knees,soles,toes
vis_items='all'
radius=8

python vis_preds.py ${img_dir} ${vis_dir} --path-contact-states=${save_path} --joint-names=${joint_names} --vis-items=${vis_items} --radius=${radius}
