## Training contact recognizer


### Prepare datasets

Download and set up training data by running:
```terminal
wget https://www.di.ens.fr/willow/research/motionforcesfromvideo/contact-recognizer/training_data.tar.gz
tar -xf training_data.tar.gz -C ~/contact-recognizer/data/joint_images/
rm training_data.tar.gz
```
Our training set consists of fixed-size image patches of human joints with manually annotated ground-truth contact states.
The joint images are obtained by cropping from Internet images and video frames capturing human-object interactions in the following two scenarios:
- People manipulating stick-like hand tools such as hammer, shovel, mop, either at home or outside.
- People performing parkour techniques. These images are mainly from the original [LAAS Parkour MoCap Database](https://gepettoweb.laas.fr/parkour/), from videos which are not included in the [LAAS Parkour Dataset](https://github.com/zongmianli/Parkour-dataset).

In some cases, for example, when trying to recognize contact under a new scenario, it is useful to fine-tune the model on new data in order to tackle with the domain shift.
To this end, we propose a pipeline for collecting new images and videos, annotating contact states, and generating the standard HDF5 data files required by the training/testing modules.
Detailed instructions can be found in [doc/create_data.md](https://github.com/zongmianli/contact-recognizer/blob/public/doc/create_data.md).


### Training

Now we can start training.
We use [scripts/train.sh](https://github.com/zongmianli/contact-recognizer/blob/public/scripts/train.sh) as a sample script to train a *contact recognizer for hands*.
The same approach applies to other human joints as well.
Update the parameters in the beginning of the script and run:
```
source scripts/train.sh
```
This script will call the training module [main.py](https://github.com/zongmianli/contact-recognizer/blob/public/main.py) with the default configuration.
Please read the code for more details.
