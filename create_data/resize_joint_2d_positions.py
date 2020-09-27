import argparse
import cPickle as pk
import numpy as np
import numpy.linalg as LA
from os import makedirs
from os.path import join, exists, abspath, dirname, basename


if __name__ == '__main__':
    '''
    Rescale joint 2D positions according to a set of scale factors.
    '''
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('path_datainfo', type=str, metavar='DIR',
                        help='Path to a data_info.pkl file.')
    parser.add_argument('path_j2d', type=str, metavar='DIR',
                        help='Path to the corresponding Openpose outputs.')
    parser.add_argument('path_resizeinfo', type=str, metavar='DIR',
                        help='Path to the scale factors.')
    parser.add_argument('save_path', type=str, metavar='DIR',
                        help='Output path.')

    args = parser.parse_args()
    path_datainfo = args.path_datainfo
    path_j2d = args.path_j2d
    path_resizeinfo = args.path_resizeinfo
    save_path = args.save_path

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

    # Load scale factors for all items
    with open(path_resizeinfo, 'r') as f:
        data_resizeinfo = pk.load(f)
        item_scales = data_resizeinfo['item_scales']
        if (not len(data_resizeinfo["item_names"]) == len(item_names)):
            raise ValueError('check failed: len(data_resizeinfo["item_names"]) == len(item_names)')

    print("Resizing joints in {0:s} ...".format(path_j2d))
    new_joint_positions = dict()
    for i, item_name in enumerate(item_names):
        item_scale = item_scales[i]
        print("  #{0:d} {1:s}: scale {2:2.2f}".format(i, item_name, item_scale))
        j2d_pos = joint_positions_dict[item_name]
        j2d_pos_new = j2d_pos.copy()
        j2d_pos_new[:,:,:2] = item_scale*j2d_pos[:,:,:2]
        new_joint_positions[item_name] = j2d_pos_new

    if not exists(dirname(save_path)):
        makedirs(dirname(save_path))
    with open(save_path, 'w') as f:
        pk.dump(new_joint_positions, f)
    print("Scaled joint 2D positions saved to {0:s}".format(save_path))
