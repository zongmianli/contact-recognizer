import argparse
import h5py
import cPickle as pk
import numpy as np
import numpy.linalg as LA
from os import makedirs
from os.path import join, exists, abspath, dirname, basename
from PIL import Image


def get_indices(joint_names):
    '''
    Given a list of joint names as input, the function retrieves their indices
    in the Openpose output array and in the contact_states array.
    '''
    # Reference info
    joint_names_all = [
        "neck", "l_hand", "r_hand", "hands",
        "l_knee", "r_knee", "knees",
        "l_sole", "r_sole", "soles",
        "l_toes", "r_toes", "toes"
    ]
    openpose_ids_all = [
        [1],[7],[4],[7,4],
        [12],[9],[12,9],
        [13],[10],[13,10],
        [13],[10],[13,10]
    ]
    contact_ids_all = [
        [0],[1],[2],[1,2],
        [3],[4],[3,4],
        [5],[6],[5,6],
        [7],[8],[7,8]
    ]
    openpose_ids = []
    contact_ids = []
    for joint in joint_names:
        # Sanity check
        if joint in joint_names_all:
            i = joint_names_all.index(joint)
            openpose_ids.append(openpose_ids_all[i])
            contact_ids.append(contact_ids_all[i])
        else:
            raise ValueError('check failed: unknown joint {0:s}'.format(joint))

    return openpose_ids, contact_ids

def count_undetected_joints(contact_states, contact_ids):
    '''
    Count the number of detected joint in contact_states for initializing hdf5 database.
    '''
    contact_states_size = (9,7)
    # Round 1: sanity check
    for i, contact_item in enumerate(contact_states):
        # Dimensions in contact_states muct match
        if contact_item.shape[1:] != contact_states_size:
            raise ValueError('check failed: contact_states[{0:d}].shape[1:] != {1}'.format(
                i, contact_states_size))

    # Round 2: scan for items missing contact annotations
    items_with_unlabelled_data = []
    for i, contact_item in enumerate(contact_states):
        number_unlabelled_joints = [0]*len(contact_ids)
        for k in range(contact_item.shape[0]):
            # Find unlabelled joints
            for n, joint_ids in enumerate(contact_ids):
                for j, j_ctt in enumerate(joint_ids): # j_ctt: joint id in contact_states
                    if contact_item[k][j_ctt].tolist() == [0]*7:
                        number_unlabelled_joints[n] += 1
        if number_unlabelled_joints != [0]*len(contact_ids):
            items_with_unlabelled_data.append(i)

    # Round 3: count the number undetected joints (items in items_with_unlabelled_data are skipped)
    contact_states_sum = np.zeros(contact_states_size).astype(int)
    num_images = 0
    for i, contact_item in enumerate(contact_states):
        if i in items_with_unlabelled_data:
            continue

        # Add up contact_item for all images
        nimages_item = contact_item.shape[0]
        for k in range(nimages_item):
            contact_states_sum += contact_item[k]
        num_images += nimages_item

    # Finally, we count the number of detected joints
    number_undetected_joints = np.zeros(len(contact_ids)).astype(int)
    for n, joint_ids in enumerate(contact_ids):
        for j_ctt in joint_ids:
            number_undetected_joints[n] += contact_states_sum[j_ctt,-1]

    return number_undetected_joints, num_images, items_with_unlabelled_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Create datasets for training contact recognizers. These are labelled joint images saved in HDF5 format.")
    parser.add_argument('image_folder', type=str, metavar='DIR',
                        help="path to image folder.")
    parser.add_argument('save_folder', type=str, metavar='DIR',
                        help="Path to the directory for saving the HDF5 data files.")
    parser.add_argument('--crop-sizes', type=str, default='80,100,120',
                        help="crop sizes expressed in pixel")
    parser.add_argument('--joint-names', type=str, default='all',
                        help="Joint names split by comma, for example, 'l_hand,r_hand,l_knee,r_knee,l_sole,r_sole'."
                        "It is also possible to merge the left and the right joints by removing the r_ or l_,"
                        "for example, 'hands,knees,soles'.")

    args = parser.parse_args()

    image_folder = args.image_folder
    save_folder = args.save_folder
    crop_sizes = [int(s) for s in args.crop_sizes.split(',')]
    joint_names = args.joint_names.split(',')

    # ---------------- Load necessary info ----------------
    path_datainfo = join(image_folder, "data_info.pkl")
    path_j2d = join(image_folder, "Openpose-video.pkl")
    path_contact_states = join(image_folder, "contact_states_annotation.pkl")
    # Sanity check
    if not exists(path_datainfo):
        raise ValueError('check failed: path_datainfo')
    if not exists(path_j2d):
        raise ValueError('check failed: path_j2d')
    if not exists(path_contact_states):
        raise ValueError('check failed: path_contact_states')

    # Load data info
    print("Load data info from {0:s}".format(path_datainfo))
    with open(path_datainfo, 'r') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        image_to_itemframe = data_info["image_to_itemframe"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]

    # Load joint 2d positions
    print("Load joint 2D locations from {0:s}".format(path_j2d))
    with open(path_j2d, 'r') as f:
        joint_positions_dict = pk.load(f)
        # Convert joint_positions_dict (dict with item_name+arrays) to joint_positions (list of arrays)
        joint_positions = [None] * len(item_names)
        for i, item_name in enumerate(item_names):
            joint_positions[i] = joint_positions_dict[item_name]

    # load contact states
    print('Load contact states from {0:s}'.format(path_contact_states))
    with open(join(path_contact_states), 'r') as f:
        data = pk.load(f)
        contact_states = data['contact_states']

    # ---------------- Crop joint patches ----------------
    # Get joint ids and contact ids
    openpose_ids, contact_ids = get_indices(joint_names)

    # Count the total number of each label in contact_states
    number_undetected_joints, num_images, item_ids_to_skip = count_undetected_joints(
        contact_states, contact_ids)

    # Ignore items with unlabelled contact states
    item_names_to_skip = []
    if len(item_ids_to_skip) > 0:
        print("Warning: Skipped *{0:d}* items with unlabelled contact states!".format(
            len(item_ids_to_skip)))
        for i in item_ids_to_skip:
            item_names_to_skip.append(item_names[i])
            print("  - {0:s}".format(item_names[i]))

    # Initialize hdf5 data file
    if not exists(save_folder):
        makedirs(save_folder)

    n_crops = len(crop_sizes)
    hf = [None] * (n_crops*len(joint_names))
    number_data = np.zeros(len(joint_names)).astype(int)
    for n, joint_name in enumerate(joint_names):
        for m, crop_size in enumerate(crop_sizes):
            hf_path = join(save_folder, "{0:s}_{1:d}.h5".format(
                joint_name, crop_size))
            hf[n*n_crops+m] = h5py.File(hf_path, 'w')
            #hf[n][m] = h5py.File(hf_path, 'w')
            if joint_name in ["hands", "knees", "soles", "toes"]:
                number_data[n] = 2*num_images - number_undetected_joints[n]
            else:
                number_data[n] = num_images - number_undetected_joints[n]
            hf[n*n_crops+m].create_dataset("images", (number_data[n], crop_size, crop_size, 3), dtype=np.uint8)
            hf[n*n_crops+m].create_dataset("contact_states", (number_data[n], 7), dtype=np.uint8)
            hf[n*n_crops+m].create_dataset("item_ids", (number_data[n],), dtype=np.uint8)
            hf[n*n_crops+m].create_dataset("frame_ids", (number_data[n],), dtype=np.uint8)
            hf[n*n_crops+m].create_dataset("contact_ids", (number_data[n],), dtype=np.uint8)
            hf[n*n_crops+m].create_dataset("openpose_ids", (number_data[n],), dtype=np.uint8)

    # Loop over item_names: crop joint images
    j_counts = np.zeros(len(joint_names)).astype(int)
    for i, item_name in enumerate(item_names):
        if item_name in item_names_to_skip:
            # Sanity check
            if item_names.index(item_name) not in item_ids_to_skip:
                raise ValueError('check failed: item_names.index(item_name) in item_ids_to_skip')
            continue

        item_len = item_lengths[i]
        if item_len==1:
            print("Processing item {0:d}/{1:d} (image): {2:s} ...".format(i, len(item_names), item_name))
        elif item_len > 1:
            print("Processing item {0:d}/{1:d} (video): {2:s} ({3:d} frames) ...".format(i, len(item_names), item_name, item_len))
        else:
            raise ValueError('item_len <= 0!')

        imgs_count = 0
        j_counts_item = np.zeros(len(joint_names)).astype(int)
        for k in range(item_len):
            # Load image
            img_path = join(image_folder, image_names[item_to_image[i] + k])
            img = Image.open(img_path)
            w, h = img.size
            # Load joint 2d positions at item i, frame k
            j2d_pos = joint_positions[i][k]
            ctt_states = contact_states[i][k]
            # Loop over all types of joints
            for n, joint_ids in enumerate(contact_ids):
                joint_ids_openpose = openpose_ids[n]
                for j, j_ctt in enumerate(joint_ids): # j_ctt: joint id in contact_states
                    # Skip if the joint is labelled as 'undetected'
                    # Crop joint images otherwise
                    if ctt_states[j_ctt, -1] == 1:
                        continue
                    else:
                        j_pose = joint_ids_openpose[j] # j_pose: joint id in joint_positions
                        center = j2d_pos[j_pose, :2] # location of the joint, center of cropping
                        for m, crop_size in enumerate(crop_sizes):
                            x, y = center
                            dx = dy = crop_size/2
                            crop_box = [x-dx, y-dy, x-dx+crop_size, y-dy+crop_size]
                            cropped = img.crop(crop_box) # crop image
                            #
                            data_images = hf[n*n_crops+m].get("images")
                            data_images[j_counts[n]] = np.array(cropped)[:,:,:3] # convert to array and write to data file
                            data_labels = hf[n*n_crops+m].get("contact_states")
                            data_labels[j_counts[n]] = ctt_states[j_ctt].copy()
                            data_item_ids = hf[n*n_crops+m].get("item_ids")
                            data_item_ids[j_counts[n]] = i
                            data_frame_ids = hf[n*n_crops+m].get("frame_ids")
                            data_frame_ids[j_counts[n]] = k
                            data_contact_ids = hf[n*n_crops+m].get("contact_ids")
                            data_contact_ids[j_counts[n]] = j_ctt
                            data_openpose_ids = hf[n*n_crops+m].get("openpose_ids")
                            data_openpose_ids[j_counts[n]] = j_pose

                        j_counts[n] += 1
                        j_counts_item[n] += 1

        if item_len > 1:
            for n, joint_name in enumerate(joint_names):
                print("  - cropped {0:d} {1:s} images ".format(j_counts_item[n], joint_name))

    # Sanity check
    if not np.array_equal(j_counts.astype(int), number_data):
        print("Warning: np.array_equal(j_counts.astype(int), number_detected_joints) is *False*: {0} vs {1}".format(j_counts.astype(int), number_data))

    # Clean up
    for n, joint_name in enumerate(joint_names):
        for m, crop_size in enumerate(crop_sizes):
            hf[n*n_crops+m].close()
