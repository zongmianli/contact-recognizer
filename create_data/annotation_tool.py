import argparse
import glob
import cv2 as cv
import numpy as np
import cPickle as pk
from os import makedirs
from os.path import join, exists, abspath, dirname, basename
# ploting modules
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# plot in notebook cell outputs
from IPython.display import clear_output



class AnnotationTool:
    '''
    Load and display a set of images with human 2d pose estimation results
    (stick figures). Then an annotator annotate the contact state of the
    subject's key joints through several predefined commands.

    The predefined commands are:
    1: joint is in contact
    2: joint is not in contact
    3: joint is occluded
    4: joint is detected and is correct
    5: joint is detected but incorrect: flipped limbs
    6: joint is detected but incorrect: wrong detection
    a: the rest of the joints are the same as the previous image.
    wq: save the annotation to file and quit
    q!: quit without saving

    labels:
    1: joint is in contact
    2: joint is not in contact
    3: joint is occluded
    4: joint is detected and is correct
    5: joint is detected but incorrect: flipped limbs
    6: joint is detected but incorrect: wrong detection
    7: joint is not detected
    Note that 1,2,3 and 4,5,6,7 are two different groups of labels, which means each joint has two labels simultaneously.

    The key joints are annotated in the following order:
    1. Neck,
    2. Left hand,
    3. Right hand,
    4. Left knee,
    5. Right knee,
    6. Left sole,
    7. Right sole
    '''
    def __init__(self, image_folder, j2dvis_folder, path_j2d):
        self.name = basename(image_folder)
        self.image_folder = image_folder
        self.j2dvis_folder = j2dvis_folder
        # load image info
        info_path = join(image_folder, 'data_info.pkl')
        with open(info_path, 'r') as f:
            data_info = pk.load(f)
            self.image_names = data_info["image_names"]
            self.image_to_itemframe = data_info["image_to_itemframe"]
            self.item_names = data_info["item_names"] # item_names: a list of image/video names
            self.item_lengths = data_info["item_lengths"] # source_length: a list of the lengths of each source
            self.item_to_image = data_info["item_to_image"] # item_to_image: a list of indices of first image of each source (i.e. video or image)
            self.num_items = len(self.item_names) # num_items: number of videos and still images
            self.num_images = len(self.image_names)
        # load 2d joint positions
        with open(path_j2d, 'r') as f:
            self.joint_positions_dict = pk.load(f) # keys are image and video names

        if not self.num_items == len(self.joint_positions_dict.keys()):
            raise ValueError("check failed: self.num_items == len(self.joint_positions_dict.keys())!")
        print("{0:d} items ({1:d} images) to annotate".format(self.num_items, self.num_images))
        # Names and Openpose ids of the joints to consider
        self.joint_ids = [1, 7, 4, 12, 9, 13, 10, 13, 10]
        self.joint_names = ['neck',
                            'l_hand',
                            'r_hand',
                            'l_knee',
                            'r_knee',
                            'l_sole',
                            'r_sole',
                            'l_toes',
                            'r_toes']
        # Names of the joints to annotate
        self.joints_to_annotate = ['neck', 'l_hand', 'r_hand', 'l_knee', 'r_knee', 'l_sole', 'r_sole', 'l_toes', 'r_toes']
        #self.joints_to_annotate = ['l_hand', 'r_hand', 'l_sole', 'r_sole', 'l_toes', 'r_toes']
        # predefined annotation commands
        self.commands = ['1','2','3','4','5','6','a','wq','q!',
                         '14','15','16','24','25','26','34','35','36']

        # Initialize contact_states with zeros arrays
        self.contact_states = [None] * self.num_items
        for n in range(self.num_items):
            nimgs_item = self.item_lengths[n]
            self.contact_states[n] = np.zeros((
                nimgs_item, len(self.joint_names), 7)).astype(int)

    def start_annotation(self,
                         item_id=0,
                         img_id=0,
                         patch_size=120,
                         stride=1):
        # id of the image which we start to annotate
        #item_id = 0 if start_from_item is None else start_from_item
        #img_id = 0 if start_from_img is None else start_from_img

        nimg_done = 0
        nj_auto = 0
        nj_anno = 0
        quit = False
        save_quit = False
        saved = False
        for n in range(item_id, self.num_items):
            source_name = self.item_names[n]

            nimgs_item = self.item_lengths[n]
            j2d_item = self.joint_positions_dict[source_name]
            contact_item = self.contact_states[n].copy()
            for fid in range(img_id, nimgs_item, stride):
                img_path = join(self.image_folder, self.image_names[self.item_to_image[n] + fid])
                j2dvis_path = img_path.replace(self.image_folder, self.j2dvis_folder)
                img = cv.imread(img_path)
                j2dvis = cv.imread(j2dvis_path)
                next_img = False
                for jname in self.joints_to_annotate:
                    j = self.joint_names.index(jname)
                    print("batch name: {:s}".format(self.name))
                    print("source {0:d}/{1:d}: {2:s}, img {3:d}/{4:d}: {5:s}".format(
                        n+1, self.num_items, source_name, fid+1, nimgs_item, img_path))
                    jid = self.joint_ids[j]
                    jpos = j2d_item[fid, jid, :].copy()
                    # mark joints that are not detected
                    if jpos[2] < 1e-4:
                        contact_item[fid, j, -1] = 1
                        nj_auto += 1
                        continue
                    # load and display original image on the left and pose estimation
                    f, axarr = plt.subplots(1, 2)
                    f.set_size_inches((20, 10))
                    # display bounding box to help to locate the joint
                    recs = [patches.Rectangle(
                        (jpos[0]-patch_size/2, jpos[1]-patch_size/2), patch_size, patch_size, fill=False,
                        edgecolor="red", linestyle='dashed') for r in range(2)]
                    # plot human 2d pose estimation results
                    axarr[0].set_title(jname+' (2D joints)')
                    axarr[0].imshow(j2dvis[:,:,[2,1,0]])
                    axarr[0].add_patch(recs[0])
                    axarr[0].axis('off')
                    # show original image
                    axarr[1].set_title(jname+' (original image)')
                    axarr[1].imshow(img[:,:,[2,1,0]])
                    axarr[1].add_patch(recs[1])
                    axarr[1].axis('off')

                    plt.show(block=False)

                    ask_cmd = True
                    while ask_cmd:
                        label = raw_input(self.joint_names[j]+
                            '('+str(int(jpos[0]))+', '+str(int(jpos[1]))+'):')
                        if label not in self.commands:
                            print 'unknown command. Enter again:'
                            continue
                        elif label=='a' and fid==img_id:
                            print 'cannot use command <a> at the first image of each item!'
                            continue
                        elif label=='a':
                            # find the index of jname in self.joints_to_annotate
                            index_start = self.joints_to_annotate.index(jname)
                            # loop over the joints to annotate
                            for idx in range(index_start, len(self.joints_to_annotate)):
                                jname = self.joints_to_annotate[idx]
                                k = self.joint_names.index(jname)
                                jid = self.joint_ids[k]
                                jpos = j2d_item[fid, jid, :].copy()
                                if jpos[2] < 1e-4:
                                    contact_item[fid, k, -1] = 1
                                    nj_auto += 1
                                else:
                                    contact_item[fid, k, :] = \
                                        contact_item[fid-stride, k, :].copy()
                                    # contact_item[fid, k, 3:] = \
                                    #     contact_item[fid-stride, k, 3:].copy()
                                    nj_anno += 1
                            clear_output()
                            ask_cmd = False
                            next_img = True
                        elif label=='wq':
                            # write to file and quit
                            self.contact_states[n] = contact_item.copy()
                            self.set_annotation_zero(n, fid)
                            clear_output()
                            ask_cmd = False
                            save_quit = True
                        elif label=='q!':
                            # quit
                            self.set_annotation_zero(item_id, img_id)
                            clear_output()
                            ask_cmd = False
                            quit = True
                        else:
                            for l in label:
                                contact_item[fid, j, int(l)-1] = 1
                            nj_anno += 1
                            ask_cmd = False
                            clear_output()

                    plt.close()

                    if next_img or quit or save_quit:
                        break

                if quit or save_quit:
                    break
                nimg_done += 1
            if quit or save_quit:
                break
            self.contact_states[n] = contact_item.copy()

        if not quit:
            saved = self.save_annotations()

        if saved:
            print 'you have annotated {} images,'.format(nimg_done)
            print 'with {} detected (and annotated) joints.'.format(nj_anno)
            print 'with {} undetected joints.'.format(nj_auto)
            if save_quit:
                print 'you stopped at source: #{0:d}, image: #{1:d}'.format(n, fid)
                print 'begin with these item and image ids next time!'
            else:
                print 'images from '+self.image_folder+' are all annotated'
                print 'well done!'
        else:
            print 'quit without saving'

    def set_annotation_zero(self, item_id, img_id):
        '''
        set contact annotation to zero for items>=item_id, images>=img_ig
        '''
        for n in range(item_id, self.num_items):
            nimgs_item = self.item_lengths[n]
            contact_item = self.contact_states[n]
            if contact_item is not None:
                img_id_start = 0
                if n==item_id:
                    img_id_start = img_id
                for fid in range(img_id_start, nimgs_item):
                    self.contact_states[n][fid] = np.zeros((len(self.joint_names), 7))

    def save_annotations(self):
        save_path = join(self.image_folder, 'contact_states_annotation.pkl')
        data = {}
        data['name'] = self.name
        data['num_items'] = self.num_items
        data['item_names'] = self.item_names
        data['num_images'] = self.num_images
        data['joints_positions_dict'] = self.joint_positions_dict
        data['joint_names'] = self.joint_names
        data['joint_ids'] = self.joint_ids
        data['contact_states'] = self.contact_states
        with open(save_path, 'w') as f:
            pk.dump(data, f)
            print('annotation saved to {:s}'.format(save_path))
            return True

    def load_annotations(self, anno_path):
        with open(join(anno_path), 'r') as f:
            data = pk.load(f)
            print('annotation load from {:s}'.format(anno_path))
            self.name = data['name']
            self.image_folder = join(dirname(self.image_folder), self.name)
            self.num_items = data['num_items']
            self.joint_positions_dict = data['joints_positions_dict']
            self.joint_names = data['joint_names']
            self.joint_ids = data['joint_ids']
            self.contact_states = data['contact_states']

    def check_all(self, contact_states, stride=1):

        # Get joint_positions from self.joint_positions_dict
        joint_positions = [None] * self.num_items
        for n in range(self.num_items):
            source_name = self.item_names[n]
            joint_positions[n] = self.joint_positions_dict[source_name]

        # Check contact state annotations for all items
        num_items = len(contact_states)
        for n in range(num_items):
            source_name = self.item_names[n]
            raw_input("Processing item #{0:d} ({1:s}). Press Enter to continue ...".format(n, source_name))
            contact_item = contact_states[n]
            j2d_item = joint_positions[n]
            if contact_item is None:
                print("*** item {0:d} is missing".format(n))
            else:
                nimgs_item = contact_item.shape[0]
                for fid in range(0, nimgs_item, stride):
                    if self.check_single_image(contact_item[fid], j2d_item[fid]):
                        print("  in item {0:d}/{1:d}, img {2:d}/{3:d}".format(
                            n, num_items-1, fid, nimgs_item-1))
        print('done.')

    def check_single_source(self, contact_source, stride=1):
        # Warning: this function doesn't compare contact states with joint confidences
        nimgs_item = contact_source.shape[0]
        for fid in range(0, nimgs_item, stride):
            if self.check_single_image(contact_source[fid]):
                print("  in img {0:d}/{1:d}".format(fid, nimgs_item-1))

    def check_single_image(self, mat_contact, mat_j2d=None):
        num_labels = mat_contact.shape[1]
        report = False
        for jname in self.joints_to_annotate:
            j = self.joint_names.index(jname)

            # Make sure all entries are either 0 or 1
            for k in range(num_labels):
                if mat_contact[j,k] not in [0,1]:
                    print("* Unvalid number in j=={0:d}, k=={1:d}!".format(j, k))
                    print("  Only 0 and 1 are valid inputs.")
                    report = True

            # Make sure the contact states (first three entries) are labelled
            if mat_contact[j,0]==mat_contact[j,1]==mat_contact[j,2]==0:
                print("* Unlabelled data in j=={0:d}, k==0:3!".format(j))
                report = True
            elif np.sum(mat_contact[j,:3]) != 1:
                print("* row does not sum to 1 at j=={0:d}, k==0:3!".format(j))
                report = True

            # Make sure the Openpose states (namely, are the joint detections correct
            # or not, in the last four entries) are labelled
            if mat_contact[j,3]==mat_contact[j,4]==mat_contact[j,5]==mat_contact[j,6]==0:
                print("* Unlabelled data in j=={0:d}, k==3:end!".format(j))
                report = True
            elif np.sum(mat_contact[j,3:]) != 1:
                print("* row does not sum to 1 at j=={0:d}, k==3:end!".format(j))
                report = True

            # Compare contact states with joint confidences
            if mat_j2d is not None:
                jid = self.joint_ids[j]
                if mat_j2d[jid, -1] > 1e-4 and mat_contact[j,-1] == 1:
                    print("*** detected joint labelled as non-detected (j={0:d})".format(j))
                elif mat_j2d[jid, -1] <= 1e-4 and mat_contact[j,-1] == 0:
                    print("*** undetected joint labelled as detected (j={0:d})".format(j))

        return report


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str, metavar='DIR')
    parser.add_argument('j2dvis_folder', type=str, metavar='DIR')
    parser.add_argument('path_j2d', type=str, metavar='DIR')
    args = parser.parse_args()

    image_folder = args.image_folder
    j2dvis_folder = args.j2d_folder
    path_j2d = args.path_j2d

    annot = AnnotationTool(image_folder, j2dvis_folder, path_j2d)

    annot.start_annotation(item_id=0, img_id=0, patch_size=120, stride=1)
