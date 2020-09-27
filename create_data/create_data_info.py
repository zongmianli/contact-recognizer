import argparse
import glob
import cPickle as pk
import numpy as np
import numpy.linalg as LA
from os import makedirs
from os.path import join, exists, abspath, dirname, basename

def grab_images(image_folder, image_types, sort=False):
    '''
    Grab images from image_folder, according to their image_types.
    '''
    image_paths = []
    for type_name in image_types:
        print("Grab images (*.{0:s}) from {1:s} ... ".format(type_name, image_folder))
        paths = glob.glob(join(image_folder, "*.{0:s}".format(type_name)))
        if sort:
            paths = sorted(paths)
        image_paths.extend(paths)
    return image_paths

def grab_subfolders(root_folder, sort=False):
    '''
    Grab subfolders in root_folder.
    '''
    print("Grab subfolders from {0:s} ... ".format(root_folder))
    subfolders = glob.glob(join(root_folder, '*/'))
    # Remove the '/' symbols at the end of the paths
    subfolders = [folder[:-1] for folder in subfolders]
    if sort:
        subfolders = sorted(subfolders)
    return subfolders


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str, metavar='DIR',
                        help="Path to an image folder.")
    parser.add_argument('--image_types', type=str, default='jpg,png',
                        help="image types (split by commas)")
    parser.add_argument("--save_info", default=False, action="store_true",
                        help="Save the resulting data_info.")

    args = parser.parse_args()
    image_folder = args.image_folder
    image_types = args.image_types.split(',')
    save_info = args.save_info

    # Initialization
    image_names = []
    image_to_itemframe = []
    item_names = [] # image/video names that are used to search in joint_positions
    item_to_image = [] # indices from which each item starts in image_names and scales
    item_lengths = [] # length of each item: 1 for still image, number of frames for video.

    # Grab still images
    image_paths = grab_images(image_folder, image_types, sort=False)
    image_names.extend([basename(path) for path in image_paths])
    image_to_itemframe.extend([[i, 0] for i in range(len(image_names))])
    item_names.extend([basename(path) for path in image_paths])
    item_to_image.extend(range(len(image_names)))
    item_lengths.extend([1] * len(image_names))

    # Grab video frame paths and compute scale factors
    video_folders = grab_subfolders(image_folder, sort=True)
    for i, video_folder in enumerate(video_folders):
        video_name = basename(video_folder)

        frame_paths = grab_images(video_folder, image_types, sort=True)
        num_frames = len(frame_paths)
        image_names.extend(["{0:s}/{1:s}".format(video_name, basename(path)) for path in frame_paths])
        image_to_itemframe.extend([[len(item_names), k] for k in range(num_frames)])

        item_names.append(video_name)
        if len(item_to_image)>0 and len(item_lengths)>0:
            item_to_image.append(item_to_image[-1]+item_lengths[-1])
            item_lengths.append(num_frames)
        else:
            item_to_image.append(0)
            item_lengths.append(num_frames)

    data_info = {
        "image_names": image_names,
        "image_to_itemframe": image_to_itemframe,
        "item_names": item_names,
        "item_lengths": item_lengths,
        "item_to_image": item_to_image
    }

    if save_info:
        with open(join(image_folder, "data_info.pkl"), 'w') as f:
            pk.dump(data_info, f)
