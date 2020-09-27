import argparse
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
from train_model import train_model
from evaluate_model import evaluate_model
from utils.utils import load_parameters_from_txt


def rank_model(models_folder,
               parameters_id,
               hf_name,
               mean_ap,
               acc,
               max_len=10):
    '''
    Ranking and saving top-N best checkpoints in terms of mAP and accuracy.
    '''

    models_dir = join(models_folder, hf_name)
    checkpoint_dir = join(models_dir, "checkpoints")
    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)

    # Randomly sample a checkpoint id of length 8
    checkpoint_id = "{0:032x}".format(random.getrandbits(128))[:8]

    # Make sure the sampled id has not been used, otherwise sample a new one
    ranking_path = join(models_dir, 'model_rankings.pkl')
    if exists(ranking_path):
        with open(ranking_path, 'r') as f:
            ranking_dict = pk.load(f)
        while True:
            accept_id = True
            checkpoint_id = "{0:032x}".format(random.getrandbits(128))[:8]
            for model_eval in ranking_dict['models']:
                if model_eval['checkpoint_id'] == checkpoint_id:
                    accept_id = False
            if accept_id:
                break

    save_name = "{0}_{1}_map{2:0.2f}_acc{3:0.4f}".format(
        parameters_id, checkpoint_id, mean_ap, acc)
    checkpoint_path = join(models_dir, 'checkpoints', save_name+'.pth.tar')
    model_eval = {'save_name':save_name,
                  'parameters_id':parameters_id,
                  'checkpoint_id':checkpoint_id,
                  'hf_name':hf_name,
                  'acc':acc,
                  'mean_ap':mean_ap}

    save_checkpoint = False
    # Get ranking lists
    ranking_dict = dict()
    if exists(ranking_path):
        with open(ranking_path, 'r') as f:
            ranking_dict = pk.load(f)
        # if model_eval is better than any of the models in the ranking lists
        # append model_eval to the lists and set save_checkpoint to True
        # if the list length larger than max_len, then remove the worst model,
        # from both the ranking list and models list
        list_names = ['acc', 'mean_ap']
        values = [acc, mean_ap]
        list_rms_cache = []
        for name, value in zip(list_names, values):
            rank_list = ranking_dict[name]
            list_len = len(rank_list)
            for i in range(list_len):
                if value>rank_list[i][2]:
                    rank_list.insert(i, (parameters_id, checkpoint_id, value))
                    if not save_checkpoint:
                        ranking_dict['models'].append(model_eval)
                        save_checkpoint = True
                    break
                if i==list_len-1 and list_len<max_len:
                    rank_list.append((parameters_id, checkpoint_id, value))
                    if not save_checkpoint:
                        ranking_dict['models'].append(model_eval)
                        save_checkpoint = True
                    break

            if len(rank_list)>max_len:
                ckptid_rm = rank_list[-1][1]
                ranking_dict[name] = rank_list[:max_len]
                # Add checkpoint id to remove list if it is not in the list
                if not any([ckptid_rm==s for s in list_rms_cache]):
                    list_rms_cache.append(ckptid_rm)

        # Check if checkpoints in list_rms_cache appear in other lists
        list_rms = []
        for ckptid_rm in list_rms_cache:
            rm_confirm = True
            for name in list_names:
                rank_list = ranking_dict[name]
                for i in range(len(rank_list)):
                    if ckptid_rm==rank_list[i][1]:
                        rm_confirm = False
                        break
            if rm_confirm:
                list_rms.append(ckptid_rm)

        if len(list_rms)>0:
            # Update models key in ranking_dict
            models_update = []
            for modeldict in ranking_dict['models']:
                if any([modeldict['checkpoint_id']==s for s in list_rms]):
                    # Remove the checkpoint from system
                    rm_path = join(models_dir, 'checkpoints',
                                   modeldict['save_name']+'.pth.tar')
                    remove(rm_path)
                else:
                    models_update.append(modeldict)
            ranking_dict['models'] = models_update
    else:
        ranking_dict = {
            'models': [model_eval],
            'acc': [(parameters_id, checkpoint_id, acc)],
            'mean_ap': [(parameters_id, checkpoint_id, mean_ap)]}
        save_checkpoint = True

    if save_checkpoint:
        with open(ranking_path, 'w') as fout:
            pk.dump(ranking_dict, fout)

    return save_checkpoint, checkpoint_id, checkpoint_path


def main(experiment_dir,
         job_name,
         paths_training_data,
         strides_training_data,
         paths_validation_data,
         strides_validation_data,
         models_folder,
         resume,
         novis,
         joint_name,
         num_epochs,
         parameters_id):

    # ------------------------------------------------------------------
    # Sanity check

    hf_name = basename(paths_training_data[0]).split('.')[0]
    patch_size = int(hf_name.split('_')[-1])

    for n in range(1, len(paths_training_data)):
        if hf_name != basename(paths_training_data[n]).split('.')[0]:
            raise ValueError(
                "Should never happen: "
                "hf_name != basename(paths_training_data[n]).split('.')[0] "
                "({0:s} vs {1:s})".format(
                hf_name, basename(paths_training_data[n]).split('.')[0]))

    if joint_name != hf_name[:len(joint_name)]:
        raise ValueError(
            "Should never happen: "
            "joint_name == hf_name[:len(joint_name)] ({0:s} vs {1:s})".format(
            joint_name, hf_name[:len(joint_name)]))

    if not len(paths_training_data)==len(strides_training_data):
        raise ValueError(
            "Should never happen: "
            "len(paths_training_data)==len(strides_training_data) "
            "({0:d} vs {1:d})".format(
            len(paths_training_data), len(strides_training_data)))

    if not len(paths_validation_data)==len(strides_validation_data):
        raise ValueError(
            "Should never happen: "
            "len(paths_validation_data)==len(strides_validation_data) "
            "({0:d} vs {1:d})".format(
            len(paths_validation_data), len(strides_validation_data)))

    print("(main.py) Parameters & settings ...")
    print(" - joint_name: {0}".format(joint_name))
    print(" - patch_size: {0}".format(patch_size))
    print(" - parameters_id: {0}".format(parameters_id))
    print(" - experiment_dir: {0}".format(experiment_dir))
    print(" - job_name: {0}".format(job_name))
    print(" - num_epochs: {0}".format(num_epochs))


    # ------------------------------------------------------------------
    epoch_resume = -1
    if resume:
        print("(main.py) Resuming checkpoint ...\n"
              " - Resume path: {0:s}".format(resume))
        checkpt = torch.load(resume)
        joint_name = checkpt['joint_name']
        patch_size = checkpt['patch_size']
        parameters_id = checkpt['parameters_id']
        epoch_resume = checkpt['epoch']
        nclasses_resume = checkpt['nclasses']
        loss_record_resume = checkpt['loss_record']
        acc_record_resume = checkpt['acc_record']
        model_state_resume = checkpt['model_state']
        optim_state_resume = checkpt['optimizer_state']

        print("(main.py) Resumed parameters & settings:")
        print(" - joint_name: {0}".format(joint_name))
        print(" - patch_size: {0}".format(patch_size))
        print(" - parameters_id: {0}".format(parameters_id))
        print(" - epoch: {0}".format(epoch_resume))
        print(" - nclasses: {0}".format(nclasses_resume))


    # ------------------------------------------------------------------
    print("(main.py) Loading parameters and settings ...")

    params = load_parameters_from_txt(
        "parameters/{0:s}.txt".format(parameters_id))
    batch_size = params['batch_size'][0]
    freeze_depth = params['freeze_depth'][0]
    lr = params['lr'][0]
    lr_decay_stepsize = params['lr_decay_stepsize'][0]
    momentum = params['momentum'][0]
    weight_decay = params['weight_decay'][0]
    rotdegree = params['rotdegree'][0]
    hflip = params['hflip'][0]
    brightness = params['brightness'][0]
    contrast = params['contrast'][0]
    saturation = params['saturation'][0]
    hue = params['hue'][0]

    print(" - parameters_id: {0}".format(parameters_id))
    print(" - batch_size: {0}".format(batch_size))
    print(" - freeze_depth: {0}".format(freeze_depth))
    print(" - lr: {0}".format(lr))
    print(" - lr_decay_stepsize: {0}".format(lr_decay_stepsize))
    print(" - momentum: {0}".format(momentum))
    print(" - weight_decay: {0}".format(weight_decay))
    print(" - rotdegree: {0}".format(rotdegree))
    print(" - hflip: {0}".format(hflip))
    print(" - brightness: {0}".format(brightness))
    print(" - contrast: {0}".format(contrast))
    print(" - saturation: {0}".format(saturation))
    print(" - hue: {0}".format(hue))


    # ------------------------------------------------------------------
    # Set up data augmentation and normalization for training data,
    # and normalization for validating data
    print("(main.py) Setting up data transforms ...")

    train_transforms = [
        transforms.Resize(size=(224,224)),
        transforms.RandomRotation(
            degrees=rotdegree, resample=False, expand=False, center=None),
        transforms.ColorJitter(brightness=brightness,
                               contrast=contrast,
                               saturation=saturation,
                               hue=hue),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if hflip == "true":
        train_transforms.insert(0, transforms.RandomHorizontalFlip())
    val_transforms = [
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(val_transforms)}


    # ------------------------------------------------------------------
    # Create training and validation sets
    print("(main.py) Creating training and validation sets ...")

    if joint_name == "neck":
        label_scheme = 1
    else:
        label_scheme = 3

    image_datasets = dict()
    phase_names = []

    # Split the training data to train and val sets if the path
    # to validation data is not provided
    create_train_val_info = False
    if len(paths_validation_data)==1 and paths_validation_data[0]=='':
        create_train_val_info = True

    if not create_train_val_info:
        image_datasets['train'] = ContactDataset(
            paths_training_data,
            hf_strides=strides_training_data,
            label_scheme=label_scheme,
            subset_items=None,
            transform=data_transforms['train'])
        phase_names.append('train')
        classes = np.unique(image_datasets['train'].labels).tolist()
        nclasses = len(classes)
        image_datasets['val'] = ContactDataset(
            paths_validation_data,
            hf_strides=strides_validation_data,
            label_scheme=label_scheme,
            subset_items=None,
            transform=data_transforms['val'])
        phase_names.append('val')
    else:
        print(" - Split training data into training and validation sets ...")
        path_train_val_split = join(
            experiment_dir, "train_val_split_{}.pkl".format(joint_name))

        if exists(path_train_val_split):
            with open(path_train_val_split, 'r') as f:
                data = pk.load(f)
                label_scheme = data["label_scheme"]
                item_ids_train = data["item_ids_train"]
                item_ids_val = data["item_ids_val"]
                classes = data["classes"]
                print(" - Train-val split info read from:\n"
                  "    - {0:s}".format(path_train_val_split))
        else:
            print(" - Saving train val split info to:\n"
                  "   {0:s}".format(path_train_val_split))
            image_dataset = ContactDataset(
                paths_training_data,
                hf_strides=strides_training_data,
                label_scheme=label_scheme,
                subset_items=None,
                transform=None)

            train_val_ratio = 4.
            item_ids_all = np.unique(image_dataset.item_ids)
            nitems_all = len(item_ids_all)
            nitems_train = int(nitems_all*train_val_ratio/(train_val_ratio+1))
            order = range(nitems_all)
            random.shuffle(order)
            item_ids_train = item_ids_all[order[:nitems_train]]
            item_ids_val = item_ids_all[order[nitems_train:]]

            classes = np.unique(image_dataset.labels).tolist()

            data = dict()
            data["label_scheme"] = label_scheme
            data["item_ids_train"] = item_ids_train
            data["item_ids_val"] = item_ids_val
            data["classes"] = classes
            with open(path_train_val_split, 'w') as f:
                pk.dump(data, f)

        # Initialize train and validation datasets
        image_datasets['train'] = ContactDataset(
            paths_training_data,
            hf_strides=strides_training_data,
            label_scheme=label_scheme,
            subset_items=item_ids_train,
            transform=data_transforms['train'])
        phase_names.append('train')

        image_datasets['val'] = ContactDataset(
            paths_training_data,
            hf_strides=strides_training_data,
            label_scheme=label_scheme,
            subset_items=item_ids_val,
            transform=data_transforms['val'])
        phase_names.append('val')
        nclasses = len(classes)


    # ------------------------------------------------------------------
    print("(main.py) Initializing dataloaders ...")

    dataset_sizes = dict()
    dataloaders = dict()
    for phase in phase_names:
        dataset_sizes[phase] = len(image_datasets[phase])
        print(" - {0:s} set : {1:d} joint images".format(
            phase, dataset_sizes[phase]))

        if phase == "train":
            # Use weighted sampler for unbalanced dataset
            labels_train = image_datasets[phase].labels
            labels_train_bin = label_binarize(labels_train, classes=range(3))
            labels_train_bin = labels_train_bin[:,:nclasses]
            prob = np.sum(
                labels_train_bin, axis=0).astype(float)/dataset_sizes[phase]
            prob = np.clip(prob, 0.05, 0.9)

            reciprocal_weights = np.zeros(dataset_sizes[phase])
            for i in range(dataset_sizes[phase]):
                reciprocal_weights[i] = prob[classes.index(labels_train[i])]
            weights = torch.DoubleTensor((1. / reciprocal_weights).tolist())
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights, dataset_sizes[phase], replacement=True)

            dataloaders[phase] = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=0)
        else:
            dataloaders[phase] = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0)


    # ------------------------------------------------------------------
    print("(main.py) Loading model ...")

    use_gpu = torch.cuda.is_available()

    model_ft = models.resnet18()
    if resume:
        print(" - Resuming model state ...")
        if not nclasses == nclasses_resume:
            raise ValueError(
                "Should never happen: "
                "nclasses == nclasses_resume ({0} vs {1})".format(
                nclasses, nclasses_resume))
        model_ft.fc = nn.Linear(model_ft.fc.in_features, nclasses)
        model_ft.load_state_dict(model_state_resume)
    else:
        model_path = join(models_folder, "resnet18-5c106cde.pth")
        print(" - {0:s}".format(model_path))
        model_ft.load_state_dict(torch.load(model_path))
        # Adjust output number
        model_ft.fc = nn.Linear(model_ft.fc.in_features, nclasses)


    # ------------------------------------------------------------------
    print("(main.py) Freezing top layers ...")
    if freeze_depth not in [0,5,6,7,8]:
        raise ValueError("Invalide freeze_depth number!")
    child_counter = 0
    for child in model_ft.children():
        if child_counter < freeze_depth:
            for param in child.parameters():
                param.requires_grad = False
            print " - child {} was frozen".format(child_counter)
        else:
            print " - child {} was not frozen".format(child_counter)
        child_counter += 1


    # ------------------------------------------------------------------
    print("(main.py) Initializing optimizer ...")
    if use_gpu:
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(
            filter(lambda p: p.requires_grad, model_ft.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)

    # Resume optimizer state
    if resume:
        print(" - Optimizer state resumed")
        optimizer_ft.load_state_dict(optim_state_resume)
    else:
        print(" - Optimizer state initialized from scratch")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Gradually decay lr during optimization
    exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft,
            step_size=lr_decay_stepsize,
            gamma=0.1,
            last_epoch=-1)


    # ------------------------------------------------------------------
    print("(main.py) Start training ...")
    best_epoch = -1
    acc_record = {x: np.zeros((0)) for x in phase_names}
    loss_record = {x: np.zeros((0)) for x in phase_names}

    model_ft, optimizer_ft, best_epoch, loss_record, acc_record = \
        train_model(
            dataloaders,
            dataset_sizes,
            phase_names,
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs,
            use_gpu)

    if resume:
        # Count the previous epochs
        best_epoch = epoch_resume + best_epoch + 1
        # Concatenate loss_record and acc_record
        for phase in phase_names:
            acc_record[phase] = np.concatenate(
                (acc_record_resume[phase], acc_record[phase]))
            loss_record[phase] = np.concatenate(
                (loss_record_resume[phase], loss_record[phase]))


    # ------------------------------------------------------------------
    # Evaluate estimation accuracy, precision-recall and ROC

    acc_by_phase, precision_by_phase, recall_by_phase, ap_by_phase, tpr_by_phase, fpr_by_phase, roc_auc_by_phase, scores_by_phase, labels_by_phase = \
        evaluate_model(joint_name,
                       dataloaders,
                       classes,
                       phase_names,
                       dataset_sizes,
                       model_ft,
                       use_gpu)

    # Compute mAPs from ap_by_phase
    map_by_phase = dict()
    ignore_labels = [2] # ignore the "occluded" label for computing mAP
    for phase in phase_names:
        map_effective = []
        for n in range(nclasses):
            if classes[n] in ignore_labels:
                continue
            name = str(classes[n])
            map_effective.append(ap_by_phase[phase][name])
        map_by_phase[phase] = np.mean(np.array(map_effective))

    # Rank the model according to accuracy and mAP, save the top-10
    # best models as checkpoints
    save_checkpoint, checkpoint_id, checkpoint_path = rank_model(
        models_folder,
        parameters_id,
        hf_name,
        map_by_phase["val"],
        acc_by_phase["val"],
        max_len=10)

    if save_checkpoint:
        acc_record_save = \
            {x: acc_record[x][:(best_epoch+1)] for x in phase_names}
        loss_record_save = \
            {x: loss_record[x][:(best_epoch+1)] for x in phase_names}
        data_dict = {
            'checkpoint_id': checkpoint_id,
            'parameters_id': parameters_id,
            'nclasses' : nclasses,
            'epoch': best_epoch,
            'loss_record': loss_record_save,
            'acc_record': acc_record_save,
            'model_state': model_ft.state_dict(),
            'optimizer_state': optimizer_ft.state_dict(),
            'paths_training_data': paths_training_data,
            'paths_validation_data': paths_validation_data,
            'joint_name': joint_name,
            'patch_size': patch_size,
        }

        torch.save(data_dict, checkpoint_path)
        print("(main.py) Checkpoint saved to: \n - {0:s}".format(
            checkpoint_path))

    # ------------------------------------------------------------------
    # Save training statistics and evaluation results
    results_path = join(experiment_dir, job_name+'_results.pkl')
    results_data = dict()
    if exists(results_path):
        with open(results_path, 'r') as f:
            results_data = pk.load(f)
    else:
        for phase in phase_names:
            results_data["map-"+phase] = []
            results_data["acc-"+phase] = []
        results_data["checkpoint_ids"] = []
        results_data["parameters_ids"] = []

    for phase in phase_names:
        results_data["map-"+phase].append(map_by_phase[phase])
        results_data["acc-"+phase].append(acc_by_phase[phase])
    results_data["checkpoint_ids"].append(checkpoint_id)
    results_data["parameters_ids"].append(parameters_id)

    with open(results_path, 'w') as f:
        pk.dump(results_data, f)
        print("(main.py) Training statistics and evaluation results "
              "saved to: \n - {0:s}".format(results_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Training contact recognizer")
    parser.add_argument(
        'experiment_dir', type=str,
        help="Path to a directory for saving results and plots")
    parser.add_argument(
        'job_name', type=str, help="Job name")
    parser.add_argument(
        'paths_training_data', type=str,
        help="Paths to *.h5 files for training, strings split by comma")
    parser.add_argument(
        'strides_training_data', type=str,
        help="Strides for loading training data, integers split by comma")
    parser.add_argument(
        '--paths-validation-data', type=str, default='',
        help="Paths to *.h5 files for validation, strings split by comma. "
        "This argument is useful when users want to fix the validation set. "
        "By default the training program will randomly split the training "
        "data into train and validation sets")
    parser.add_argument(
        '--strides-validation-data', type=str, default='1',
        help="Strides for loading validation data, integers split by comma")
    parser.add_argument(
        '--models-folder', type=str, default='models', metavar='PATH',
        help="Path to a directory for saving model checkpoints")
    parser.add_argument(
        '--resume', type=str, default='', metavar='PATH',
        help="Path to a checkpoint to resume")
    parser.add_argument(
        '--novis', default=False, action='store_true',
        help="No visualization")
    parser.add_argument(
        '--joint-name', type=str, default='hands',
        help="Specify joint name")
    parser.add_argument(
        '--num-trials', type=int, default=1,
        help="Number of trials")
    parser.add_argument(
        '--num-epochs', type=int, default=10,
        help="Planned number of epochs")
    parser.add_argument(
        '--parameters-id', type=str, default='00000000',
        help="Name of the *.txt containing parameters and settings")

    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    job_name = args.job_name
    paths_training_data = args.paths_training_data.split(',')
    strides_training_data = \
        [int(s) for s in args.strides_training_data.split(',')]
    paths_validation_data = args.paths_validation_data.split(',')
    strides_validation_data = \
        [int(s) for s in args.strides_validation_data.split(',')]
    models_folder = args.models_folder
    resume = args.resume
    novis = args.novis
    joint_name = args.joint_name
    num_trials = args.num_trials
    num_epochs = args.num_epochs
    parameters_id = args.parameters_id

    for n in range(num_trials):
        # ------------------------------------------------------------------
        print("(main.py) ============== training trial {0:2d} / {1:2d} "
              "==============".format(n+1, num_trials))

        main(experiment_dir,
             job_name,
             paths_training_data,
             strides_training_data,
             paths_validation_data,
             strides_validation_data,
             models_folder,
             resume,
             novis,
             joint_name,
             num_epochs,
             parameters_id)
