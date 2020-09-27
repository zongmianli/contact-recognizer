import argparse
import cPickle as pk
import numpy as np
import numpy.linalg as LA
from os import makedirs
from os.path import join, exists, abspath, dirname, basename
from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from create_data.create_data import get_indices


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Visualizing joint contact states on images")
    parser.add_argument('image_folder', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('--path-contact-states', type=str, default='')
    parser.add_argument(
        '--joint-names', type=str, default='hands,soles',
        help="Joint names split by comma, for example, 'hands,knees,soles'.")
    parser.add_argument(
        '--vis-items', type=str, default='all',
        help="Names of the items to proceed")
    parser.add_argument(
        '--radius', type=int, default=5,
        help="Radius of the dots showing joint location and contact states")

    args = parser.parse_args()
    image_folder = args.image_folder
    save_folder = args.save_folder
    path_contact_states = args.path_contact_states
    joint_names = args.joint_names.split(',')
    vis_items = args.vis_items.split(',')
    radius = args.radius

    # ------------------------------------------------------------------
    # Load basic info
    path_datainfo = join(image_folder, "data_info.pkl")
    path_j2d = join(image_folder, "Openpose-video.pkl")
    if not path_contact_states:
        path_contact_states = join(
            dirname(image_folder), "contact_states_annotation.pkl")

    # Sanity check
    if not exists(path_datainfo):
        raise ValueError('check failed: path_datainfo')
    if not exists(path_j2d):
        raise ValueError('check failed: path_j2d')
    if not exists(path_contact_states):
        raise ValueError('check failed: path_contact_states')

    # ------------------------------------------------------------------
    # Load data info
    print("(vis_preds.py) Loading image folder info ...\n"
          " - Path: {0:s}".format(path_datainfo))

    with open(path_datainfo, 'r') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        image_to_itemframe = data_info["image_to_itemframe"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]

    if args.vis_items == 'all':
        vis_items = item_names

    # ------------------------------------------------------------------
    # Load joint 2d positions
    print("(vis_preds.py) Loading joint 2D positions ...\n"
          " - Path: {0:s}".format(path_j2d))

    with open(path_j2d, 'r') as f:
        joint_positions_dict = pk.load(f)
        # Convert joint_positions_dict (dict with item_name+arrays)
        # to joint_positions (list of arrays)
        joint_positions = [None] * len(item_names)
        for i, item_name in enumerate(item_names):
            joint_positions[i] = joint_positions_dict[item_name]

    # ------------------------------------------------------------------
    # Load contact states
    print("(vis_preds.py) Loading contact states ...\n"
          " - Path: {0:s}".format(path_contact_states))

    with open(join(path_contact_states), 'r') as f:
        data = pk.load(f)
        contact_states = data['contact_states']
        # Sanity check
        if not data['item_names']==item_names:
            print("data['item_names'] == ")
            print(data['item_names'])
            print("item_names == ")
            print(item_names)
            raise ValueError("check failed: data['item_names']==item_names")

    # ---------------- Visualize contact states ----------------
    # Get joint ids and contact ids
    openpose_ids, contact_ids = get_indices(joint_names)

    # Colors
    colors = {
        0: "green",
        1: "red",
        2: "orange"
    }

    if not exists(save_folder):
        makedirs(save_folder)

    # Loop over all items and images
    for i, item_name in enumerate(item_names):
        if item_name not in vis_items:
            continue

        item_len = item_lengths[i]
        if item_len==1:
            print("(vis_preds.py) Processing item {0:d}/{1:d} (image)"
                  ": {2:s} ...".format(i, len(item_names), item_name))
        elif item_len > 1:
            print("(vis_preds.py) Processing item {0:d}/{1:d} (video)"
                  ": {2:s} ({3:d} frames) ...".format(
                      i+1, len(item_names), item_name, item_len))
            if not exists(join(save_folder, item_name)):
                makedirs(join(save_folder, item_name))
        else:
            raise ValueError('item_len <= 0!')

        nclasses = contact_states[i].shape[2] - 1

        for k in range(item_len):
            print(" - image #{0}".format(k))
            # Load image
            img_name = image_names[item_to_image[i] + k]
            img_path = join(image_folder, img_name)
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            w, h = img.size
            # Load joint 2d positions at item i, frame k
            j2d_pos = joint_positions[i][k]
            ctt_states = contact_states[i][k]

            # Loop over all types of joints
            for n, joint_ids in enumerate(contact_ids):
                joint_ids_openpose = openpose_ids[n]
                for j, j_ctt in enumerate(joint_ids):
                    # j_ctt: joint id in contact_states
                    # Skip undetected joints
                    # Crop joint images otherwise
                    j_pose = joint_ids_openpose[j]
                    # j_pose: joint id in joint_positions
                    x, y = j2d_pos[j_pose, :2].astype(int) # joint location
                    conf = j2d_pos[j_pose, 2] # confidence score
                    if conf < 1e-4:
                        continue

                    # sanity check
                    if not len(ctt_states[j_ctt]) == nclasses+1:
                        raise ValueError(
                            "check failed: len(ctt_states[j_ctt])"
                            " == nclasses+1 ({0} vs {1})".format(
                                len(ctt_states[j_ctt]), nclasses+1))
                    if not np.sum(ctt_states[j_ctt][:-1]) == 1:
                        raise ValueError(
                            "check failed: np.sum(ctt_states[j_ctt][:-1])"
                            " == 1")

                    label = ctt_states[j_ctt].tolist().index(1)

                    if j_ctt in [5,6]: # soles
                        draw.ellipse(
                            (x-radius, y-radius, x+radius, y+radius),
                            fill = None,
                            outline = colors[label],
                            width=radius/2)
                    elif j_ctt in [7,8]:
                        draw.ellipse(
                            (x-radius/2, y-radius/2, x+radius/2, y+radius/2),
                            fill = colors[label],
                            outline = None)
                    else:
                        draw.ellipse(
                            (x-radius, y-radius, x+radius, y+radius),
                            fill = colors[label],
                            width=2)

            # Crop image to enforce even height & width.
            # (required by ffmpeg if we want to convert to videos)
            w_target = w/2*2
            h_target = h/2*2
            if w!=w_target or h!=h_target:
                crop_box = [0, 0, w_target, h_target]
                img = img.crop(crop_box)

            save_path = join(save_folder, img_name)
            img.save(save_path)
