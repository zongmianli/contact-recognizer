#!/bin/bash

for name in full_images joint_images
do
    wget https://www.di.ens.fr/willow/research/motionforcesfromvideo/contact-recognizer/demo_data_${name}.tar.gz
    mkdir -p ~/contact-recognizer/data/${name}
    tar -xf demo_data_${name}.tar.gz -C ~/contact-recognizer/data/${name}/
    rm demo_data_${name}.tar.gz
done
