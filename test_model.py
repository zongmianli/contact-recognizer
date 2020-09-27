import argparse
import copy
import random
import h5py
import time
import cPickle as pk
import numpy as np
from datetime import datetime
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

from contact_dataset import ContactDataset


def test_model(dataloader, dataset_size, model_ft, nclasses, use_gpu):

    print(' - (test_model.py) Start testing ...')

    since = time.time()

    # Set model_ft to evaluate mode
    model_ft.train(False)

    labels_pred = np.zeros(dataset_size)
    scores_pred = -1 * np.ones((0,nclasses))

    # Iterate over data.
    idx = 0
    batch_size = dataloader.batch_size

    for inputs, _ in dataloader:
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)

        if use_gpu:
            labels_pred[idx:(idx+batch_size)] = preds.cpu().numpy()
            scores_pred = np.concatenate((
                scores_pred, outputs.cpu().data.numpy()), axis=0)
        else:
            labels_pred[idx:(idx+batch_size)] = preds.numpy()
            scores_pred = np.concatenate((
                scores_pred, outputs.data.numpy()), axis=0)
        idx += batch_size

    time_elapsed = time.time() - since
    print(' - (test_model.py) Testing took {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return labels_pred.astype(int), scores_pred


def main(joint_name, resume, data_path, info_path, save_path):

    # ------------------------------------------------------------------
    print("(test_model.py) Resuming checkpoint ...\n"
          " - Resume path: {0:s}".format(resume))
    checkpt = torch.load(resume)
    joint_name = checkpt['joint_name']
    patch_size = checkpt['patch_size']
    parameters_id = checkpt['parameters_id']
    epoch_resume = checkpt['epoch']
    nclasses_resume = checkpt['nclasses']
    model_state_resume = checkpt['model_state']

    print("(test_model.py) Resumed parameters & settings:")
    print(" - joint_name: {0}".format(joint_name))
    print(" - patch_size: {0}".format(patch_size))
    print(" - parameters_id: {0}".format(parameters_id))
    print(" - epoch: {0}".format(epoch_resume))
    print(" - nclasses: {0}".format(nclasses_resume))


    # ------------------------------------------------------------------
    print("(test_model.py) Setting up data transforms ...")

    data_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # ------------------------------------------------------------------
    # Create testing set
    # In testing-mode, all joint images are loaded without label
    print("(test_model.py) Setting up dataloader for test set ...")

    test_dataset = ContactDataset(
        [data_path],
        hf_strides=[1],
        label_scheme=None,
        subset_items=None,
        transform=data_transform)

    testset_size = len(test_dataset)
    testset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0)

    # ------------------------------------------------------------------
    print("(test_model.py) Loading model ...")

    # Load pretrained convnet
    model_ft = models.resnet18()
    # Change output number
    model_ft.fc = nn.Linear(model_ft.fc.in_features, nclasses_resume)
    # Load model
    model_ft.load_state_dict(model_state_resume)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    # ------------------------------------------------------------------

    # Load image info
    with open(info_path, 'r') as f:
        data_info = pk.load(f)
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        num_items = len(item_names)

    # Load Openpose ids and contact ids from data file
    hf = h5py.File(data_path, 'r')
    item_ids = hf.get("item_ids")[()]
    frame_ids = hf.get("frame_ids")[()]
    contact_ids = hf.get("contact_ids")[()]
    hf.close()

    # Predict labels
    labels_pred, scores_pred = test_model(
        testset_loader, testset_size, model_ft, nclasses_resume, use_gpu)

    # Initialize contact_states_pred and scores
    scores = [None] * num_items
    contact_states_pred = [None] * num_items
    if exists(save_path):
        with open(save_path, 'r') as f:
            data = pk.load(f)
            # Sanity check
            if not num_items==len(data["scores"]):
                raise ValueError(
                    "check failed: "
                    "num_items==len(data['scores']) ({0} vs {1})".format(
                        num_items, len(data["scores"])))
            if not num_items==len(data["contact_states"]):
                raise ValueError(
                    "check failed: "
                    "num_items==len(data['contact_states']) "
                    "({0} vs {1})".format(
                        num_items, len(data["contact_states"])))

            scores = data["scores"]
            contact_states_pred = data["contact_states"]
    else:
        joint_names = ['neck',
                       'l_hand', 'r_hand',
                       'l_knee', 'r_knee',
                       'l_sole', 'r_sole',
                       'l_toes', 'r_toes']
        for n in range(num_items):
            nimgs_item = item_lengths[n]

            # Initialize scores
            scores[n] = np.zeros((
                nimgs_item, len(joint_names), 4)).astype(float)
            scores[n][:,:,-1] = 1. # initialize all scores as "undetected"

            # Initialize predicted contact states
            contact_states_pred[n] = scores[n].copy().astype(int)

    # Convert labels to contact_states_pred
    for n in range(testset_size):
        i = item_ids[n]
        k = frame_ids[n]
        j_ctt = contact_ids[n]

        # Update contact states array
        # Four labels: contact, not in contact, occluded, undetected
        pred = labels_pred[n]
        ctt_state = np.zeros(4).astype(int)
        ctt_state[pred] = 1
        contact_states_pred[i][k][j_ctt] = ctt_state

        # Update scores
        scores[i][k][j_ctt,:] = 0. # Set the corresponding score back to zero
        scores[i][k][j_ctt,:nclasses_resume] = scores_pred[n].copy()

    data = dict()
    data["contact_states"] = contact_states_pred
    data["scores"] = scores
    data["item_names"] = item_names
    with open(save_path, 'w') as f:
        pk.dump(data, f)
        print("(test_model.py) Contact states saved to: \n - {0:s}".format(
            save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Testing contact recognizer")
    parser.add_argument(
        'joint_name', type=str,
        help="name of the joint of interest")
    parser.add_argument(
        'resume', type=str,
        help="path to the checkpoint to resume")
    parser.add_argument(
        'data_path', type=str,
        help="name of the hdf5 file containing test data")
    parser.add_argument(
        'info_path', type=str,
        help="path to data_info.pkl (in image folder)")
    parser.add_argument(
        '--save-path', type=str, default=None)

    args = parser.parse_args()
    joint_name = args.joint_name
    resume = args.resume
    data_path = args.data_path
    info_path = args.info_path
    save_path = args.save_path

    # ------------------------------------------------------------------
    print("(test_model.py) ============== testing mode (joint name: {0:s})"
          "==============".format(joint_name))

    if save_path is None:
        save_path = join(dirname(info_path), "contact_states_pred.pkl")

    main(joint_name, resume, data_path, info_path, save_path)
