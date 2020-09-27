## Testing contact recognizer


### Prepare datasets

We test on two public datasets:
- [Handtool dataset](https://github.com/zongmianli/Handtool-dataset)
- [The LAAS Parkour dataset](https://github.com/zongmianli/Parkour-dataset)

Download and set up the testing data by running:
```terminal
wget https://www.di.ens.fr/willow/research/motionforcesfromvideo/contact-recognizer/testing_data.tar.gz
tar -xf testing_data.tar.gz -C ~/contact-recognizer/data/joint_images/
rm testing_data.tar.gz
```
The testing data consist of joint images cropped from the Handtool dataset and the LAAS Parkour datasets.
The joint images are saved in the HDF5 files in `parkour_test/` and `handtool_test/`.

To test on your own data, it is suggested to follow the instructions in [doc/create_data.md](https://github.com/zongmianli/contact-recognizer/blob/public/doc/create_data.md) to generate a proper HDF5 dataset of joint images.

### Testing

Please refer to [scripts/run_demo.sh](https://github.com/zongmianli/contact-recognizer/blob/public/scripts/run_demo.sh).
