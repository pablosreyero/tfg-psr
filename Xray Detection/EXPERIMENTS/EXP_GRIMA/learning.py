""" MAIN SCRIPT TO LAUNCH RPN(region proposal networks) TRAINING EXPERIMENTS

EVERY EXPERIMENT IS COMPOSED OF THREE MAIN STAGES:
- DATA PREPARATION. Functions for that are contained in module:
        - data_lists.py.
- TRAINING. Functions for that are contained in module:
        - train_RPN.py This calls:
                -../LEARNING_COMMON/processing_data.py
                -../BACKBONE_ARCHITECTURES/
                -../IMAGENET_MODELS/
- TEST. Functions for that are contained in module:
        - test_RPN.py

EVERY EXPERIMENT IS CHARACTERISED BY:
 - its identification number N (EXP_N)
 - its base network architecture (VGG, RESNET, INCEPTION...).
    Every base architecture is saved in an independent python module
 - its config.json file
 - its directory to save all related files
"""

import sys
sys.path.insert(1, '../..')
from CODE.DATA_ENGINEERING.data_lists import *
from CODE.FASTER_RCNN_FW.train_faster import *

# ADDRESSES
# The base path contains all the data related to an experiment: training, validation and test data, learnt models,
# results, learning curves...
N = 'GRIMA'
experiment_path = '../../EXPERIMENTS/EXP_' + N

# 0. READ CONFIG FILE
config_path = os.path.join(experiment_path, 'config.json')

# 1. DATA PREPARING. DEFINING SAMPLES SETS (TRAIN, VAL, TEST)
data_class = DataLists(config_path)
data_class.samples_sets_definition_for_all_defects_GRIMA_train_test_files(True, True, 10, True)
#data_class.get_mean_ch_level_GRIMA(data_class.train_samples_gt_path)


# 2. TRAINING
#training = TrainFaster(experiment_path, config_path)
#training.training(True, False, 5, False, 2000, True, 5, True) # shuffle, explore_data, n, show_predictions, max_boxes, control_images, k, verbose

# 3. LEARNING EVALUATION
#training.draw_learning_curves()

# 4. ERROR ANALYSIS
#test = TestFaster(experiment_path, config_path)
#test.PR_curve(600, 0.5, True, True) # max_boxes, iou_th, show, verbose
#test.error_analysis(5, 600, False, True) # samples_num, max_boxes, verbose

# 5. TEST
#test.test_metrics(True, 1000, True, True)

