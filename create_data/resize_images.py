import argparse
import cPickle as pk
import numpy as np
import numpy.linalg as LA
from os import makedirs
from os.path import join, exists, abspath, dirname, basename
from PIL import Image

def compute_scale_factor(image_width, width_limits, joint_2d_positions=None):
    '''
    Compute scale factor according to image width limits and human limb lengths.
    '''

    # Resize original images according to torso and leg sizes
    scale = 1. # initialize scale to 1.
    if joint_2d_positions is not None:
        # Predefine a set of torso and leg links
        links_torso = [[5, 11], [2, 8], [5, 8], [2, 11]]
        links_legs = [[11, 12], [8, 9], [12, 13], [9, 10]]

        # Set target torso and leg lengths (in pixel)
        target_torso_len = 140.
        target_leg_len = 120.

        # Compute scale factors for each image
        scales = np.ones(len(joint_2d_positions))

        for i in range(len(joint_2d_positions)):
            j2d_pos = joint_2d_positions[i]

            # Compute the length of the longest (detected) torso limb
            max_torso_len = 0.
            for jid_pair in links_torso:
                joint_detected = j2d_pos[jid_pair, 2] > 1e-2
                link_detected = joint_detected[0] and joint_detected[1]
                if link_detected:
                    link_len = LA.norm(j2d_pos[jid_pair[0], :2] - j2d_pos[jid_pair[1], :2])
                    max_torso_len = max(max_torso_len, link_len)

            # Compute the length of the longest (detected) leg limb
            max_leg_len = 0.
            for jid_pair in links_legs:
                joint_detected = j2d_pos[jid_pair, 2] > 1e-2
                link_detected = joint_detected[0] and joint_detected[1]
                if link_detected:
                    link_len = LA.norm(j2d_pos[jid_pair[0], :2] - j2d_pos[jid_pair[1], :2])
                    max_leg_len = max(max_leg_len, link_len)

            if max_torso_len >= max_leg_len and max_torso_len > 0.:
                scales[i] = target_torso_len/max_torso_len
            elif max_leg_len > max_torso_len and max_leg_len > 0.:
                scales[i] = target_leg_len/max_leg_len

        scale = np.mean(scales)
        #raw_input("scale == {}".format(scale))

    # Recompute the scale factor if w_resized and/or h_resized are beyond their limits
    w_resized = image_width * scale
    w_min, w_max = width_limits
    if w_resized < w_min:
        scale = w_min/float(image_width)
    elif w_resized > w_max:
        scale = w_max/float(image_width)

    return scale

def resize_images(image_paths, scales, save_folder):
    '''
    Thus function reads images from image_paths, resize them according to scales,
    and save the resized images to save_folder.
    The input list images_paths and scales are supposed to have the same length.
    '''
    if not exists(save_folder):
        makedirs(save_folder)

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        w, h = img.size
        s = scales[i]
        img_resized = img.resize(((int)(w*s), (int)(h*s)), Image.ANTIALIAS)
        save_path = join(save_folder, basename(img_path))
        img_resized.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resize images according to people's sizes")
    parser.add_argument('image_folder', type=str, metavar='DIR',
                        help="Path to an image folder.")
    parser.add_argument('save_folder', type=str, metavar='DIR',
                        help="Path to the output folder for saving resized images.")
    parser.add_argument('path_j2d', type=str, metavar='DIR',
                        help='Path to the pkl file containing Openpose 2D joint locations.')
    parser.add_argument('--width_limits', type=str, metavar='MIN,MAX', default='300,1500',
                        help="min_width,max_width")
    args = parser.parse_args()

    image_folder = args.image_folder
    save_folder = args.save_folder
    path_j2d = args.path_j2d
    width_limits = [float(n) for n in args.width_limits.split(',')]

    # Load data info
    print("Load data info from {0:s}".format(join(image_folder, "data_info.pkl")))
    with open(join(image_folder, "data_info.pkl"), 'r') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        image_to_itemframe = data_info["image_to_itemframe"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]
        # Initialize scale factors for all items
        item_scales = [None] * len(item_names)

    # Load joint 2d positions
    print("Load joint 2D locations from {0:s}".format(path_j2d))
    with open(path_j2d, 'r') as f:
        joint_positions_dict = pk.load(f)

    # Compute person bbox and scale factor for each item
    print("Resizing {0:d} items, {1:d} images in {2:s} ...".format(
        len(item_names), len(image_names), image_folder))
    for i, item_name in enumerate(item_names):
        print("  #{0:d} {1:s}".format(i, item_name))
        item_length = item_lengths[i]

        # Sanity check
        if(not item_length==joint_positions_dict[item_name].shape[0]):
            raise ValueError('check failed: item_length==joint_positions_dict[item_name].shape[0]')

        # Compute the scale factor according to person's size
        # This code works on both images and video frames
        # For videos, it will load the first frame image of the video to get the frame with
        img_path = join(image_folder, image_names[item_to_image[i]])
        img = Image.open(img_path)
        item_width = img.size[0] # (width, height)
        j2d_pos = joint_positions_dict[item_name]
        item_scales[i] = compute_scale_factor(
            item_width, width_limits, joint_2d_positions=j2d_pos)

        # Resize and save new images
        scales = [item_scales[i]]*item_length
        image_paths = [None]*item_length
        for k in range(item_length):
            image_paths[k] = join(image_folder, image_names[item_to_image[i] + k])
        if item_length == 1:
            # For still images, save the resized image under save_folder
            resize_images(image_paths, scales, save_folder)
        else:
            # For videos, save frame images to a subfolder
            resize_images(image_paths, scales, join(save_folder, item_name))

    resize_info = {"item_scales": item_scales, "item_names": item_names}
    with open(join(save_folder, "resize_info.pkl"), 'w') as f:
        pk.dump(resize_info, f)

    print("done.")

