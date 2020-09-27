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


def train_model(dataloaders,
                dataset_sizes,
                phase_names,
                model_ft,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                num_epochs,
                use_gpu):
    since = time.time()

    best_model = copy.deepcopy(model_ft.state_dict())
    memo_optimizer = copy.deepcopy(optimizer_ft.state_dict())
    lowest_loss = 9999.0
    best_epoch = -1

    acc_record = {x: np.zeros(num_epochs) for x in phase_names}
    loss_record = {x: np.zeros(num_epochs) for x in phase_names}

    for epoch in range(num_epochs):
        print(' - (train_model.py) Epoch {0:d} / {1:d}'.format(
            epoch + 1, num_epochs))
        # Each epoch has training and validation phases
        for phase in phase_names:
            if phase == 'train':
                exp_lr_scheduler.step()
                model_ft.train(True)  # Set training mode
            else:
                model_ft.train(False)  # Set evaluate mode

            running_loss = 0.0
            running_corrects = 0.

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer_ft.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            loss_record[phase][epoch] = epoch_loss
            acc_record[phase][epoch] = epoch_acc

            print('    - {0:s} loss: {1:.4f}, accuracy: {2:.2f}%'.format(
                phase, epoch_loss, epoch_acc*100))

        validation_loss = loss_record['val'][epoch]
        if validation_loss < lowest_loss:
            lowest_loss = validation_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model_ft.state_dict())
            memo_optimizer = copy.deepcopy(optimizer_ft.state_dict())

    time_elapsed = time.time() - since
    print(' - (train_model.py) Training took {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print(' - (train_model.py) Lowest val_loss: {:6f}'.format(lowest_loss))

    # Load best model weights
    model_ft.load_state_dict(best_model)
    optimizer_ft.load_state_dict(memo_optimizer)
    return model_ft, optimizer_ft, best_epoch, loss_record, acc_record
