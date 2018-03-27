#! /usr/bin/env python
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json


if __name__ == '__main__':
    config_path = 'config/config.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

    print('Seen labels:\t', train_labels)
    print('Given labels:\t', config['model']['labels'])
    print('Overlap labels:\t', overlap_labels)

    if len(overlap_labels) < len(config['model']['labels']):
        print('Some labels have no images! Please revise the list of labels in the config.json file!')
    ###############################
    #   Construct the model
    ###############################
    else:
        yolo = YOLO(architecture=config['model']['architecture'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'], trainable=True)

        ###############################
        #   Load the pretrained weights (if any)
        ###############################

        if os.path.exists(config['train']['pretrained_weights']):
            print("Loading pre-trained weights in", config['train']['pretrained_weights'])
            yolo.load_weights(config['train']['pretrained_weights'])  # ########################

        ###############################
        #   Start the training process
        ###############################
        # yolo.model.load_weights(config['train']['pretrained_weights'])
        yolo.train(train_imgs=train_imgs,
                   valid_imgs=valid_imgs,
                   train_times=config['train']['train_times'],
                   valid_times=config['valid']['valid_times'],
                   nb_epoch=config['train']['nb_epoch'],
                   learning_rate=config['train']['learning_rate'],
                   batch_size=config['train']['batch_size'],
                   warmup_epochs=config['train']['warmup_epochs'],
                   object_scale=config['train']['object_scale'],
                   no_object_scale=config['train']['no_object_scale'],
                   coord_scale=config['train']['coord_scale'],
                   class_scale=config['train']['class_scale'],
                   saved_weights_name=config['train']['saved_weights_name'],
                   debug=config['train']['debug'])
        # yolo.model.save_weights('lp.h5')
