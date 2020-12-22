"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import h5py
import json
import datetime
import numpy as np
from imgaug import augmenters as iaa
import skimage.io
import re
import scipy.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import math
from keras.callbacks import LearningRateScheduler

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/diventura/")

############################################################
#  Configurations
############################################################

class DiventuraConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "diventura"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1  # Background + nucleus
    NUM_CLASSES_S = 2 + 1  # Background + nucleus
    
    # Loss type
    LOSS_TYPE = "naive"

    # Number of hypotheses
    NUM_HYPS = 1
    
    # Top-N
    TOP_N = 1

    # Number of training and validation steps per epoch

    STEPS_PER_EPOCH = 12# // IMAGES_PER_GPU

    VALIDATION_STEPS = max(1, 4 // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.85

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # maximum epochs to train

    EPOCH = 20000


    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 1.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 3000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class DiventuraInferenceConfig(DiventuraConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.3


############################################################
#  Dataset
############################################################

class DiventuraDataset(utils.Dataset):

    def load(self, dataset_dir, subset_dir):
        
        self.add_class("diventura", 1, "nucleus")
        self.add_class("diventura", 2, "cytosol")
        self.data={}

        dataset_dir = os.path.join(dataset_dir, subset_dir)
        image_ids = next(os.walk(dataset_dir))[2]
        image_ids = list(set(image_ids))

        for i,image_id in enumerate(image_ids):
            self.add_image(
                "diventura",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id))
            f = h5py.File(os.path.join(dataset_dir, image_id),'r')
            input_data=f['data'][()]
            f.close()
            self.data[i]=input_data
            
            
    def load_input(self, image_id):

        
        return self.data[image_id]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "diventura":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
############################################################
#  Training
############################################################
def exponential_step_decay(epoch):
    initial_lrate = config.LEARNING_RATE
    k = 0.01
    lrate = initial_lrate * math.exp(-epoch*k)
    return lrate

def linear_step_decay(epoch):
    initial_lrate = config.LEARNING_RATE
    final_lrate = 0.00002
    lrate = initial_lrate - (initial_lrate-final_lrate)*epoch/config.EPOCH

def cosine_decay(epoch):
    
    lrate_max = config.LEARNING_RATE
    lrate_min = 0.00000002
    epoch_max = float(config.EPOCH)
    lrate = lrate_min + 0.5 * (lrate_max - lrate_min) * (1.0 + math.cos((float(epoch) / epoch_max) * math.pi))
    return lrate

def no_decay(epoch):
    lrate = config.LEARNING_RATE
    return lrate

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = DiventuraDataset()
    dataset_train.load(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DiventuraDataset()
    dataset_val.load(dataset_dir, "val")
    dataset_val.prepare()

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur(sigma=(0.0, 0.5))
    ])
    

    lrate = LearningRateScheduler(no_decay)

    callbacks_list = [lrate]
    #print("Train all layers")
    #model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                #epochs=config.EPOCH,
                #augmentation=augmentation,
                #custom_callbacks=callbacks_list,
                #layers='all')


    #print("Train network heads")
    #model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                #epochs=config.PRE_EPOCH,
                #augmentation=augmentation,
                #layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCH,
                augmentation=augmentation,
                custom_callbacks=callbacks_list,
                layers='all')

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, epoch):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    submit_dir = "results_%s_%s" % (subset, epoch)
    submit_dir = os.path.join(config.RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = DiventuraDataset()
    dataset.load(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    avgAP = 0
    lgnd = []
    percent_ent_list = []
    percent_ent_calib_list = []
    precision_ent_list = []
    recall_ent_list = []
    f1score_ent_list = []
    percent_list = []
    precision_list = []
    recall_list = []
    f1score_list = []
    accuracy_ent_calib_list = []
    percent_ent_list2 = []
    precision_ent_list2 = []
    recall_ent_list2 = []
    f1score_ent_list2 = []
    percent_list2 = []
    precision_list2 = []
    recall_list2 = []
    f1score_list2 = []
    
    pr = 0.0
    
    eval = True
    
    match_gt = np.zeros([0], dtype=int)
    match_pred = np.zeros([0], dtype=int)
    score_pred = np.zeros([0], dtype=int)
    
    s_match_gt = []
    s_match_pred = []
    s_score_pred = []
    
    f_match_gt = [[], [], [], [], [], [], [], [], []]
    f_match_pred = [[], [], [], [], [], [], [], [], []]
    f_score_pred = [[], [], [], [], [], [], [], [], []]
    
    s_precisions = []
    s_recalls = []
    
    #import ipdb
    #ipdb.set_trace()
    
    for image_id in dataset.image_ids:
        # Load image and run detection
        #input_data = dataset.load_input(image_id)
        #[image, mask, ignore]=np.split(input_data,[1,input_data.shape[2]-1],axis=2)
        #image = skimage.color.gray2rgb(np.squeeze(image))
        
        image, image_meta, gt_class_ids, gt_boxes, gt_masks, ignore = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # Detect objects
        r = model.detect([image], verbose=0)[0]

        #net1 = '/misc/lmbraid19/cicek/DiVentura/final_aleatoric/results_' + subset + '_11000/'
        #net2 = '/misc/lmbraid19/cicek/DiVentura/final_aleatoric_2/results_' + subset + '_11000/'
        #net3 = '/misc/lmbraid19/cicek/DiVentura/final_aleatoric_3/results_' + subset + '_11000/'

        #masks = []
        #scores = []
        #class_ids = []
        
        #for i in range(r['rois'].shape[0]):
            #masks.append(r['soft_masks'][r['rois'][i][0]:r['rois'][i][2],r['rois'][i][1]:r['rois'][i][3],:,i])
            #scores.append(r['scores'][i])
            #class_ids.append(r['class_ids'][i])
            
        #scipy.io.savemat(net3 + dataset.image_info[image_id]["id"]+'.mat',{'rois':r['rois'], 'masks':masks, 'scores':scores, 'class_ids':class_ids}) 
        
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        if not source_id[3] == '.' and not source_id[3] == '_':
            idx = source_id[2] + source_id[3]
        else:
            idx = source_id[2]

        if config.LOSS_TYPE == 'naive' or config.LOSS_TYPE == 'aleatoric':
            visualize.display_instances(
                image, r['rois'], r['masks'], r['soft_masks'], r['entropies'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=False, show_mask=True,
                title="Predictions", submit_dir=submit_dir, id=idx)
            plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
            
            #contents1 = scipy.io.loadmat(net1 + dataset.image_info[image_id]["id"]+'.mat')
            
            #rois1 = contents1['rois']
            #scores1 = contents1['scores'][0]
            #class_ids1 = contents1['class_ids'][0]
            
            #masks1 = np.zeros((image.shape[0], image.shape[1], 3, contents1['masks'][0].shape[0]))
            #for i in range(contents1['masks'][0].shape[0]):
                #masks1[rois1[i][0]:rois1[i][2],rois1[i][1]:rois1[i][3],:,i] = contents1['masks'][0][i]
            
            #contents2 = scipy.io.loadmat(net2 + dataset.image_info[image_id]["id"]+'.mat')
            
            #rois2 = contents2['rois']
            #scores2 = contents2['scores'][0]
            #class_ids2 = contents2['class_ids'][0]
            
            #masks2 = np.zeros((image.shape[0], image.shape[1], 3, contents2['masks'][0].shape[0]))
            #for i in range(contents2['masks'][0].shape[0]):
                #masks2[rois2[i][0]:rois2[i][2],rois2[i][1]:rois2[i][3],:,i] = contents2['masks'][0][i]
            
            #contents3 = scipy.io.loadmat(net3 + dataset.image_info[image_id]["id"]+'.mat')
            
            #rois3 = contents3['rois']
            #scores3 = contents3['scores'][0]
            #class_ids3 = contents3['class_ids'][0]
            
            #masks3 = np.zeros((image.shape[0], image.shape[1], 3, contents3['masks'][0].shape[0]))
            #for i in range(contents3['masks'][0].shape[0]):
                #masks3[rois3[i][0]:rois3[i][2],rois3[i][1]:rois3[i][3],:,i] = contents3['masks'][0][i]
            
            #rois4 = r['rois']
            #soft_masks4 = r['soft_masks']
            #scores4 = r['scores']
            #class_ids4 = r['class_ids']
            #masks4 = r['masks']
            #entropies4 = r['entropies']
            
            #soft_masks1 = masks1
            #masks1 = np.argmax(masks1, axis=2)
            
            #soft_masks2 = masks2
            #masks2 = np.argmax(masks2, axis=2)
            
            #soft_masks3 = masks3
            #masks3 = np.argmax(masks3, axis=2)
            
            #masks, soft_masks, entropies, scores, rois, class_ids = utils.ensemble(class_ids1, class_ids2, class_ids3, class_ids4, rois1, rois2, rois3, rois4, masks1, masks2, masks3, masks4, soft_masks1, soft_masks2, soft_masks3, soft_masks4, scores1, scores2, scores3, scores4, ignore)
            
            #visualize.display_instances(
                #image, rois, masks, soft_masks, entropies, class_ids,
                #dataset.class_names, scores,
                #show_bbox=False, show_mask=True,
                #title="Predictions", submit_dir=submit_dir, id=idx)
            #plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
            
            rois = r['rois']
            soft_masks = r['soft_masks']
            scores = r['scores']
            class_ids = r['class_ids']
            masks = r['masks']
            entropies = r['entropies']
            
            if eval:
                gt_match, pred_match, overlaps, pred_score = utils.compute_matches(
                                    gt_boxes, gt_class_ids, gt_masks,
                                    rois, class_ids, scores, masks,
                                    ignore, iou_threshold=0.5, ignore=True)
                
                match_gt = np.concatenate([match_gt, gt_match])
                match_pred = np.concatenate([match_pred, pred_match])
                score_pred = np.concatenate([score_pred, pred_score])
                
                ########################################################################
                
                p, r, s_pred_match, s_gt_match, s_pred_score = utils.compute_ap_sparsify(
                            gt_boxes, gt_class_ids, gt_masks, rois, class_ids, scores, masks, 
                            soft_masks, entropies, ignore, iou_threshold=0.5)
                
                s_match_gt.append(s_gt_match)
                s_match_pred.append(s_pred_match)
                s_score_pred.append(s_pred_score)
                
                ########################################################################
                
                percent, precision, recall, f1score, percent_ent, precision_ent, recall_ent, f1score_ent = utils.compute_ap_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5, c=1)
                
                percent_ent_list = percent_ent_list +  percent_ent
                precision_ent_list = precision_ent_list + precision_ent
                recall_ent_list = recall_ent_list + recall_ent
                f1score_ent_list = f1score_ent_list + f1score_ent
                
                percent_list = percent_list + percent
                precision_list = precision_list + precision
                recall_list = recall_list + recall
                f1score_list = f1score_list + f1score
                
                ########################################################################
                
                percent2, precision2, recall2, f1score2, percent_ent2, precision_ent2, recall_ent2, f1score_ent2 = utils.compute_ap_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5, c=2)
                
                percent_ent_list2 = percent_ent_list2 + percent_ent2
                precision_ent_list2 = precision_ent_list2 + precision_ent2
                recall_ent_list2 = recall_ent_list2 + recall_ent2
                f1score_ent_list2 = f1score_ent_list2 + f1score_ent2
                
                percent_list2 = percent_list2 + percent2
                precision_list2 = precision_list2 + precision2
                recall_list2 = recall_list2 + recall2
                f1score_list2 = f1score_list2 + f1score2
                
                ########################################################################
                
                percent_ent_calib, accuracy_ent_calib = utils.compute_calibration_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5)
                
                percent_ent_calib_list = percent_ent_calib_list + percent_ent_calib
                accuracy_ent_calib_list = accuracy_ent_calib_list + accuracy_ent_calib
            
            if 0:
                AP, precision_ap, recall_ap, overlap_ap = utils.compute_ap(gt_boxes, gt_class_ids, gt_masks,
                            rois, class_ids, scores, masks, ignore, iou_threshold=0.5)
                avgAP += AP
                print('AP: %.2f' % AP)
                plt.figure(2)
                plt.plot(recall_ap, precision_ap)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                lgnd.append('%.2f' % AP)
                plt.show()
                
                percent, precision, recall, f1score, percent_ent, precision_ent, recall_ent, f1score_ent = utils.compute_ap_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, c=1)
                
                percent_ent_list.append(percent_ent)
                precision_ent_list.append(precision_ent)
                recall_ent_list.append(recall_ent)
                f1score_ent_list.append(f1score_ent)
                
                percent_list.append(percent)
                precision_list.append(precision)
                recall_list.append(recall)
                f1score_list.append(f1score)
                
                percent2, precision2, recall2, f1score2, percent_ent2, precision_ent2, recall_ent2, f1score_ent2 = utils.compute_ap_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, c=2)
                
                percent_ent_list2.append(percent_ent2)
                precision_ent_list2.append(precision_ent2)
                recall_ent_list2.append(recall_ent2)
                f1score_ent_list2.append(f1score_ent2)
                
                percent_list2.append(percent2)
                precision_list2.append(precision2)
                recall_list2.append(recall2)
                f1score_list2.append(f1score2)
                
                percent_ent_calib, accuracy_ent_calib = utils.compute_calibration_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore)
                
                percent_ent_calib_list.append(percent_ent_calib)
                accuracy_ent_calib_list.append(accuracy_ent_calib)
                
                p, r = utils.compute_ap_sparsify(gt_boxes, gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks,           
                                        entropies, ignore, iou_threshold=0.5)
                
                plt.figure(2)
                plt.step(r, p)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                lgnd.append(p[1:-4].sum())
                plt.show()
                pr = pr + ((p[1:-4].sum()) / 2.0)
            
        elif config.LOSS_TYPE == 'hyp' or config.LOSS_TYPE == 'hyp_aleatoric' or config.LOSS_TYPE == 'ewta_aleatoric' or config.LOSS_TYPE == 'ewta':
            visualize.display_instances_hyp(
                image, r['rois'], r['masks'], r['soft_masks'], r['entropies'], r['hypotheses'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=False, show_mask=True,
                title="Predictions", submit_dir=submit_dir, id=idx)
            plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
            
            rois = r['rois']
            soft_masks = r['soft_masks']
            scores = r['scores']
            class_ids = r['class_ids']
            masks = r['masks']
            entropies = r['entropies']
            
            if eval:
                gt_match, pred_match, overlaps, pred_score = utils.compute_matches(
                                    gt_boxes, gt_class_ids, gt_masks,
                                    rois, class_ids, scores, masks,
                                    ignore, iou_threshold=0.5, ignore=True)
                
                match_gt = np.concatenate([match_gt, gt_match])
                match_pred = np.concatenate([match_pred, pred_match])
                score_pred = np.concatenate([score_pred, pred_score])
                
                ########################################################################
                
                p, r, s_pred_match, s_gt_match, s_pred_score = utils.compute_ap_sparsify(
                            gt_boxes, gt_class_ids, gt_masks, rois, class_ids, scores, masks, 
                            soft_masks, entropies, ignore, iou_threshold=0.5)
                
                s_match_gt.append(s_gt_match)
                s_match_pred.append(s_pred_match)
                s_score_pred.append(s_pred_score)
                
                ########################################################################
                
                percent, precision, recall, f1score, percent_ent, precision_ent, recall_ent, f1score_ent = utils.compute_ap_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5, c=1)
                
                percent_ent_list = percent_ent_list +  percent_ent
                precision_ent_list = precision_ent_list + precision_ent
                recall_ent_list = recall_ent_list + recall_ent
                f1score_ent_list = f1score_ent_list + f1score_ent
                
                percent_list = percent_list + percent
                precision_list = precision_list + precision
                recall_list = recall_list + recall
                f1score_list = f1score_list + f1score
                
                ########################################################################
                
                percent2, precision2, recall2, f1score2, percent_ent2, precision_ent2, recall_ent2, f1score_ent2 = utils.compute_ap_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5, c=2)
                
                percent_ent_list2 = percent_ent_list2 + percent_ent2
                precision_ent_list2 = precision_ent_list2 + precision_ent2
                recall_ent_list2 = recall_ent_list2 + recall_ent2
                f1score_ent_list2 = f1score_ent_list2 + f1score_ent2
                
                percent_list2 = percent_list2 + percent2
                precision_list2 = precision_list2 + precision2
                recall_list2 = recall_list2 + recall2
                f1score_list2 = f1score_list2 + f1score2
                
                ########################################################################
                
                percent_ent_calib, accuracy_ent_calib = utils.compute_calibration_masks_all(gt_boxes,                   
                                            gt_class_ids, gt_masks, rois, class_ids, scores, masks, soft_masks, entropies, ignore, iou_threshold=0.5)
                
                percent_ent_calib_list = percent_ent_calib_list + percent_ent_calib
                accuracy_ent_calib_list = accuracy_ent_calib_list + accuracy_ent_calib
                
            if 0:
                AP, precision_ap, recall_ap, overlap_ap = utils.compute_ap(gt_boxes, gt_class_ids, gt_masks,
                            r['rois'], r['class_ids'], r['scores'], r['masks'], ignore, iou_threshold=0.5)
                avgAP += AP
                print('AP: %.2f' % AP)
                plt.figure(2)
                plt.plot(recall_ap, precision_ap)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                lgnd.append('%.2f' % AP)
                plt.show()
                
                percent, precision, recall, f1score, percent_ent, precision_ent, recall_ent, f1score_ent = utils.compute_ap_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, r['rois'], r['class_ids'], r['scores'], r['masks'], r['soft_masks'], r['entropies'], ignore, c=1)
                
                percent_ent_list.append(percent_ent)
                precision_ent_list.append(precision_ent)
                recall_ent_list.append(recall_ent)
                f1score_ent_list.append(f1score_ent)
                
                percent_list.append(percent)
                precision_list.append(precision)
                recall_list.append(recall)
                f1score_list.append(f1score)
                
                percent2, precision2, recall2, f1score2, percent_ent2, precision_ent2, recall_ent2, f1score_ent2 = utils.compute_ap_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, r['rois'], r['class_ids'], r['scores'], r['masks'], r['soft_masks'], r['entropies'], ignore, c=2)
                
                percent_ent_list2.append(percent_ent2)
                precision_ent_list2.append(precision_ent2)
                recall_ent_list2.append(recall_ent2)
                f1score_ent_list2.append(f1score_ent2)
                
                percent_list2.append(percent2)
                precision_list2.append(precision2)
                recall_list2.append(recall2)
                f1score_list2.append(f1score2)
                
                percent_ent_calib, accuracy_ent_calib = utils.compute_calibration_masks(gt_boxes,                   
                                            gt_class_ids, gt_masks, r['rois'], r['class_ids'], r['scores'], r['masks'], r['soft_masks'], r['entropies'], ignore)
                
                percent_ent_calib_list.append(percent_ent_calib)
                accuracy_ent_calib_list.append(accuracy_ent_calib)
                
                p, r = utils.compute_ap_sparsify(gt_boxes, gt_class_ids, gt_masks, r['rois'], r['class_ids'], r['scores'], r['masks'], r['soft_masks'], r['entropies'], ignore, iou_threshold=0.5)
                
                plt.figure(2)
                plt.step(r, p)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                lgnd.append(p[1:-4].sum())
                plt.show()
                pr = pr + ((p[1:-4].sum()) / 2.0)
        
    if eval:
        indices = np.argsort(score_pred)[::-1]
        match_pred = match_pred[indices]
        
        # Compute precision and recall at each prediction box step
        precisions = np.cumsum(match_pred > -1) / (np.arange(len(match_pred)) + 1)
        recalls = np.cumsum(match_pred > -1).astype(np.float32) / len(match_gt)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                    precisions[indices])
        
        print('mAP: %.2f' % (mAP))
        plt.figure(20)
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
        plt.title('PR Curve (mAP: %.2f @IOU: 0.5)' % (mAP))
        plt.savefig("{}/ap.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        
        ########################################################################################
        
        for j in range(len(s_match_pred)):
            for i in range(len(s_match_pred[j])):
                f_match_gt[i] = np.concatenate([f_match_gt[i], s_match_gt[j][i]])
                f_match_pred[i] = np.concatenate([f_match_pred[i], s_match_pred[j][i]])
                f_score_pred[i] = np.concatenate([f_score_pred[i], s_score_pred[j][i]])
        
        for i in range(len(f_match_pred)):
            if len(f_match_pred[i]) != 0:
                indices = np.argsort(f_score_pred[i])[::-1]
                f_match_pred[i] = f_match_pred[i][indices]
            
            # Compute precision and recall at each prediction box step
            if len(f_match_pred[i]) == 0:
                s_precisions.append(0.0)
                s_recalls.append(0.0)
            else:
                s_precisions.append(np.sum(f_match_pred[i] > -1).astype(np.float32) / len(f_match_pred[i]))
                s_recalls.append(np.sum(f_match_pred[i] > -1).astype(np.float32) / len(f_match_gt[i]))
            
        s_precisions = np.stack(s_precisions, axis=0)
        s_recalls = np.stack(s_recalls, axis=0)
        
        s_precisions = np.concatenate([[0], s_precisions, [0]])
        s_recalls = np.concatenate([[0], s_recalls, [1]])
        
        for i in range(len(s_precisions) - 2, -1, -1):
            s_precisions[i] = np.maximum(s_precisions[i], s_precisions[i + 1])
            
        indices = np.where(s_recalls[:-1] != s_recalls[1:])[0] + 1
        pr = np.sum((s_recalls[indices] - s_recalls[indices - 1]) *
                 s_precisions[indices])
            
        plt.figure(40)
        plt.step(s_recalls, s_precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        lgnd.append(pr)
        plt.show()
        plt.title('PR Curve (mAP: %.2f @IOU: 0.5)' % pr)
        plt.savefig("{}/pr.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        print('PR: %.2f' % pr)
        
        ######################################################################################
        
        percent_ent = np.mean(percent_ent_list, axis=0)
        precision_ent = np.mean(precision_ent_list, axis=0)
        recall_ent = np.mean(recall_ent_list, axis=0)
        f1score_ent = np.mean(f1score_ent_list, axis=0)
        
        percent = np.mean(percent_list, axis=0)
        precision = np.mean(precision_list, axis=0)
        recall = np.mean(recall_list, axis=0)
        f1score = np.mean(f1score_list, axis=0)
        
        diff = np.sum(f1score - f1score_ent, axis=0)
        print('ABSC c=1: %.3f' % diff)
        
        plt.figure(60)
        #plt.plot(percent_ent * 100.0, precision_ent, 'b-', label='precision')
        #plt.plot(percent_ent * 100.0, recall_ent, 'r-', label='recall')
        plt.plot(percent * 100.0, f1score, 'r-', label='F1 Score')
        plt.plot(percent_ent * 100.0, f1score_ent, 'g-', label='Entropy')
        plt.xlabel('% of Pixels Removed')
        plt.ylabel('F1 Score')
        plt.show()
        plt.legend()
        plt.title('Sparsification Curve with Entropy for Nuclei (with ABSC: %.3f)' % diff)
        plt.savefig("{}/pr1.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        
        ##############################################################################
        
        percent_ent2 = np.mean(percent_ent_list2, axis=0)
        precision_ent2 = np.mean(precision_ent_list2, axis=0)
        recall_ent2 = np.mean(recall_ent_list2, axis=0)
        f1score_ent2 = np.mean(f1score_ent_list2, axis=0)
        
        percent2 = np.mean(percent_list2, axis=0)
        precision2 = np.mean(precision_list2, axis=0)
        recall2 = np.mean(recall_list2, axis=0)
        f1score2 = np.mean(f1score_list2, axis=0)
        
        diff2 = np.sum(f1score2 - f1score_ent2, axis=0)
        print('ABSC c=2: %.3f' % diff2)
        
        plt.figure(80)
        #plt.plot(percent_ent * 100.0, precision_ent, 'b-', label='precision')
        #plt.plot(percent_ent * 100.0, recall_ent, 'r-', label='recall')
        plt.plot(percent2 * 100.0, f1score2, 'r-', label='F1 Score')
        plt.plot(percent_ent2 * 100.0, f1score_ent2, 'g-', label='Entropy')
        plt.xlabel('% of Pixels Removed')
        plt.ylabel('F1 Score')
        plt.show()
        plt.legend()
        plt.title('Sparsification Curve with Entropy for Cytosol (with ABSC: %.3f)' % diff2)
        plt.savefig("{}/pr2.png".format(submit_dir, dataset.image_info[image_id]["id"]))

        ##############################################################################
        
        percent_ent_calib = np.mean(percent_ent_calib_list, axis=0)
        accuracy_ent_calib = np.mean(accuracy_ent_calib_list, axis=0)
        
        plt.figure(100)
        plt.plot(percent_ent_calib * 100.0, accuracy_ent_calib * 100.0, 'g-', label='Accuracy')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.show()
        plt.title('Calibration Curve with Entropy')
        plt.savefig("{}/calib.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        
    if 0:
        print('Avg. AP: %.2f' % (avgAP/len(dataset.image_ids)))
        plt.title('PR Curve (Avg. AP: %.2f)' % (avgAP/len(dataset.image_ids)))
        plt.legend(lgnd)
        plt.savefig("{}/ap.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        
        ##############################################################################
        
        percent_ent = np.mean(percent_ent_list, axis=0)
        precision_ent = np.mean(precision_ent_list, axis=0)
        recall_ent = np.mean(recall_ent_list, axis=0)
        f1score_ent = np.mean(f1score_ent_list, axis=0)
        
        percent = np.mean(percent_list, axis=0)
        precision = np.mean(precision_list, axis=0)
        recall = np.mean(recall_list, axis=0)
        f1score = np.mean(f1score_list, axis=0)
        
        diff = np.sum(f1score - f1score_ent, axis=0)
        print('ABSC c=1: %.3f' % diff)
        
        plt.figure(20)
        #plt.plot(percent_ent * 100.0, precision_ent, 'b-', label='precision')
        #plt.plot(percent_ent * 100.0, recall_ent, 'r-', label='recall')
        plt.plot(percent * 100.0, f1score, 'r-', label='Entropy')
        plt.plot(percent_ent * 100.0, f1score_ent, 'g-', label='F1 Score')
        plt.xlabel('% of Pixels Removed')
        plt.ylabel('F1 Score')
        plt.show()
        plt.legend()
        plt.title('Sparsification Curve with Entropy with ABSC c=1: %.3f' % diff)
        plt.savefig("{}/pr1.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        
        ##############################################################################
        
        percent_ent2 = np.mean(percent_ent_list2, axis=0)
        precision_ent2 = np.mean(precision_ent_list2, axis=0)
        recall_ent2 = np.mean(recall_ent_list2, axis=0)
        f1score_ent2 = np.mean(f1score_ent_list2, axis=0)
        
        percent2 = np.mean(percent_list2, axis=0)
        precision2 = np.mean(precision_list2, axis=0)
        recall2 = np.mean(recall_list2, axis=0)
        f1score2 = np.mean(f1score_list2, axis=0)
        
        diff2 = np.sum(f1score2 - f1score_ent2, axis=0)
        print('ABSC c=2: %.3f' % diff2)
        
        plt.figure(40)
        #plt.plot(percent_ent * 100.0, precision_ent, 'b-', label='precision')
        #plt.plot(percent_ent * 100.0, recall_ent, 'r-', label='recall')
        plt.plot(percent2 * 100.0, f1score2, 'r-', label='Entropy')
        plt.plot(percent_ent2 * 100.0, f1score_ent2, 'g-', label='F1 Score')
        plt.xlabel('% of Pixels Removed')
        plt.ylabel('F1 Score')
        plt.show()
        plt.legend()
        plt.title('Sparsification Curve with Entropy with ABSC c=2: %.3f' % diff2)
        plt.savefig("{}/pr2.png".format(submit_dir, dataset.image_info[image_id]["id"]))

        ##############################################################################

        percent_ent_calib = np.mean(percent_ent_calib_list, axis=0)
        accuracy_ent_calib = np.mean(accuracy_ent_calib_list, axis=0)
        
        plt.figure(60)
        plt.plot(percent_ent_calib * 100.0, accuracy_ent_calib, 'g-', label='Accuracy')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.show()
        plt.title('Calibration Curve with Entropy')
        plt.savefig("{}/calib.png".format(submit_dir, dataset.image_info[image_id]["id"]))
    
        ##############################################################################
    
        plt.title('PR Curve:')
        plt.legend(lgnd)
        plt.savefig("{}/pr.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        print('PR: %.2f' % pr)
        
    # Save to csv file


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--loss', required=False,
                        default="naive",
                        metavar="Loss type",
                        help="Uncertainty loss type to train the model")

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    #print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DiventuraConfig()
    else:
        config = DiventuraInferenceConfig()

    if args.loss == "aleatoric":
        config.NUM_CLASSES_S = 2 * config.NUM_CLASSES_S
        config.LOSS_TYPE = "aleatoric"
        config.NUM_HYPS = 4
    elif args.loss == "hyp_aleatoric":
        config.NUM_CLASSES_S = 4 * 2 * config.NUM_CLASSES_S
        config.LOSS_TYPE = "hyp_aleatoric"
        config.NUM_HYPS = 4
    elif args.loss == "ewta_aleatoric":
        config.NUM_CLASSES_S = 4 * 2 * config.NUM_CLASSES_S
        config.LOSS_TYPE = "ewta_aleatoric"
        config.NUM_HYPS = 4
    elif args.loss == "hyp":
        config.NUM_CLASSES_S = 4 * config.NUM_CLASSES_S
        config.LOSS_TYPE = "hyp"
        config.NUM_HYPS = 4
    elif args.loss == "ewta":
        config.NUM_CLASSES_S = 4 * config.NUM_CLASSES_S
        config.LOSS_TYPE = "ewta"
        config.NUM_HYPS = 4

    config.RESULTS_DIR = args.logs

    config.display()
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
 
    if args.weights is not None:
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
            
        regex = r".*[/\\]mask\_rcnn\_[\w-]+(\d{5})\.h5"
        m = re.match(regex, weights_path)
        epoch = int(m.group(1)) - 1 + 1

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset, epoch)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
