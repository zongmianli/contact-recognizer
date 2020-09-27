import h5py
import random
import cPickle as pk
import numpy as np
import numpy.linalg as LA
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import makedirs
from os.path import join, exists, abspath, dirname, basename
from PIL import Image, ImageOps


class ContactDataset(data.Dataset):

    def __init__(self,
                 hf_paths,
                 hf_strides=None,
                 label_scheme=None,
                 subset_items=None,
                 transform=None):

        # Generate name for the dataset
        self.name = basename(hf_paths[0]).split('.')[0]
        for n in range(1, len(hf_paths)):
            self.name += "_{0:s}".format(basename(hf_paths[n]).split('.')[0])

        # Determing hf strides
        if hf_strides is None:
            hf_strides = [1 for n in range(len(hf_paths))]

        # Load data from h5 file
        raw_joint_images = []
        raw_contact_states = []
        raw_item_ids = []
        raw_frame_ids = []
        for n, hf_path in enumerate(hf_paths):
            stride = hf_strides[n]
            hf = h5py.File(hf_path, 'r')
            raw_joint_images.extend(hf.get("images")[()][::stride])
            raw_contact_states.extend(hf.get("contact_states")[()][::stride])

            # Save the original frame ids of the images from which the
            # joint image patches are cropped
            raw_frame_ids.extend(hf.get("frame_ids")[()][::stride])

            # Save the original item ids as well. When there are multiple
            # input HDF5 files, we renumber the item ids to avoid duplicates
            item_ids_temp = hf.get("item_ids")[()][::stride]
            if len(raw_item_ids)>0:
                new_id_start = max(raw_item_ids) + 1
                item_ids_temp = [new_id_start+i for i in item_ids_temp]
            raw_item_ids.extend(item_ids_temp)

            hf.close()

        if subset_items is None:
            subset_items = raw_item_ids

        # Generate contact labels and joint images if label_scheme is not None.
        # Set label_scheme to None at test time, in order to include all (unlabelled, set to 0) data.
        if label_scheme is None:
            self.joint_images = raw_joint_images
            self.labels = [0] * len(self.joint_images)
            self.item_ids = raw_item_ids
            self.frame_ids = raw_frame_ids
        else:
            raw_labels = self.compute_labels(raw_contact_states, label_scheme)
            # Remove unlabelled data (indicated by label -1)
            self.joint_images = []
            self.labels = []
            self.item_ids = []
            self.frame_ids = []
            for i, label in enumerate(raw_labels):
                item_id = raw_item_ids[i]
                if item_id in subset_items:
                    if label >= 0:
                        self.labels.append(label)
                        self.item_ids.append(item_id)
                        self.frame_ids.append(raw_frame_ids[i])
                        self.joint_images.append(raw_joint_images[i])

        self.nimgs = len(self.joint_images)
        self.nitems = len(np.unique(self.item_ids))
        self.transform = transform

        # Print info
        print(" - (contact_dataset.py) {0:d} joint images loaded from:".format(
            self.nimgs))
        for n, hf_path in enumerate(hf_paths):
            print("    - {0:s} (stride {1:d})".format(hf_path, hf_strides[n]))


    def __getitem__(self, i):
        '''
        Args:
            i (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        img = Image.fromarray(self.joint_images[i])
        label = self.labels[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return self.nimgs


    def compute_labels(self, contact_states, scheme):
        '''
        Create a mapping from training labels to contact states.
        The function accepts four predefined schemes (1-4):
        - scheme 1 (2 labels):
          0 Contact (14), 1 not in contact or occluded (24,34)
        - scheme 2 (2 labels):
          0 Contact (14), 1 otherwise (24,34,16,26,36)
        - scheme 3 (3 labels):
          0 Contact (14), 1 not in contact (24), 2 occluded (34)
        - scheme 4 (3 labels):
          0 Contact (14), 1 not in contact (24), 2 occluded or incorrect (34,16,26,36)

        -1: wrong data
        '''
        label_mapper = dict()

        if scheme == 1:
            label_mapper = {
                0: [[1,0,0,1,0,0,0]],
                1: [[0,1,0,1,0,0,0], [0,0,1,1,0,0,0]],
            }
        elif scheme == 2:
            label_mapper = {
                0: [[1,0,0,1,0,0,0]],
                1: [[0,1,0,1,0,0,0], [0,0,1,1,0,0,0], [1,0,0,0,0,1,0], [0,1,0,0,0,1,0], [0,0,1,0,0,1,0]]
            }
        elif scheme == 3:
            label_mapper = {
                0: [[1,0,0,1,0,0,0]],
                1: [[0,1,0,1,0,0,0]],
                2: [[0,0,1,1,0,0,0]]
            }
        elif scheme == 4:
            label_mapper = {
                0: [[1,0,0,1,0,0,0]],
                1: [[0,1,0,1,0,0,0]],
                2: [[0,0,1,1,0,0,0], [1,0,0,0,0,1,0], [0,1,0,0,0,1,0], [0,0,1,0,0,1,0]]
            }
        else:
            raise ValueError("check failed: scheme in 1,2,3,4")

        # Initialize labels with zeros. Here the label 0 means
        # that the joint image will not be used during training
        nimgs = len(contact_states)
        labels = [-1] * nimgs
        for i in range(nimgs):
            label_assigned = False
            for label, maps in label_mapper.iteritems():
                for mp in maps:
                    if contact_states[i].tolist() == mp:
                        labels[i] = label
                        label_assigned = True
                        break
                if label_assigned:
                    break

        return labels


    def visualize_joint_images(self,
                               grid_shape,
                               id_start=0,
                               shuffle=False,
                               axes_pad=0.07,
                               save_path=None):

        # Sanity check
        assert len(grid_shape) == 2
        num_samples = grid_shape[0]*grid_shape[1]

        # Configuration
        border_colors = {
            -1: "gray",
            0: "green",
            1: "red",
            2: "orange"
        }

        order = range(self.nimgs)
        if shuffle:
            # Randomly shuffle images
            random.shuffle(order)
        list_samples = [self.joint_images[n] for n in order[id_start:(id_start+num_samples)]]
        list_labels = [self.labels[n] for n in order[id_start:(id_start+num_samples)]]

        # Plot image grid
        fig = plt.figure(1)
        grid = ImageGrid(fig, 111, grid_shape, axes_pad=axes_pad)
        for i in range(num_samples):
            img = Image.fromarray(list_samples[i])
            # Add borders in different colors to represent the labels
            label = list_labels[i]
            grid[i].imshow(ImageOps.expand(img, border=6, fill=border_colors[label]))

            # Turn off axes:
            grid[i].axes.get_xaxis().set_visible(False)
            grid[i].axes.get_yaxis().set_visible(False)

        print("- labels: ")
        print(list_labels)
        print("- item_ids: ")
        print([self.item_ids[n] for n in order[id_start:(id_start+num_samples)]])
        print("- frame_ids: ")
        print([self.frame_ids[n] for n in order[id_start:(id_start+num_samples)]])

        if save_path is None:
            save_path = "temp/vis_{0:s}.png".format(self.name)
        fig.savefig(save_path)
        print("visualize_joint_image(): Figure saved to {0:s}".format(save_path))


if __name__ == '__main__':

    # ------ TODO: update parameters ------
    hf_paths = [
        "./data/joint_images/my_imagefolder/knees_120.h5",
        "./data/joint_images/my_imagefolder/hands_120.h5",
    ]
    hf_strides = [
        1,
        1
    ]
    save_path = "./temp/demo_joint_images.png"
    label_scheme = 4
    grid_shape = [5,10]
    id_start = 0
    shuffle = True
    # ------ end: update parameters ------


    image_dataset = ContactDataset(
        hf_paths, hf_strides, label_scheme, subset_items=None, transform=None)

    print("{0:d} joint images".format(len(image_dataset)))

    image_dataset.visualize_joint_images(
        grid_shape, id_start=id_start, shuffle=shuffle, save_path=save_path)
