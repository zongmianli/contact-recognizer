#!/bin/bash

# This script replaces all video in the input folder to frame images
# We use the "avi" and "mp4" extensions to idenfity videos.
# Other video formats are not supported atm.

img_dir=${1}

source create_data/utils/split_string.sh

source_files=($(ls ${img_dir}))
for i in $(seq 0 $(( ${#source_files[@]} - 1 )))
do
    source_file=($(split_string ${source_files[$i]} .))
    filename=${source_file[0]}
    extension=${source_file[1]}
    if [ ${#extension} -gt 0 ]
    then
        if [ ${extension} = "avi" ] || [ ${extension} = "mp4" ]
        then
            video_path="${img_dir}/${filename}.${extension}"
            echo "replace_videos_with_frameimages.sh: processing ${video_path} ..."
            # Convert video to frame images
            save_folder="${img_dir}/${filename}"
            mkdir -p ${save_folder}
            save_format="png"
            ffmpeg -i ${video_path} -vf yadif ${save_folder}/%06d.${save_format}
            # Delete the video
            rm ${video_path}
        fi
    fi
done
echo "replace_videos_with_frameimages.sh: done."
