
import argparse
import os
import numpy as np
import json
from utils.voc import parse_voc_annotation
from utils.yolo import create_yolov3_model
from utils.generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # run on GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # run on CPU

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    #   Διάβασμα εικόνων και προσημειώσεων αξιολόγησης
    valid_ints, labels = parse_voc_annotation(
        config['valid']['valid_annot_folder'], 
        config['valid']['valid_image_folder'], 
        config['valid']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32,
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    #   Φόρτωση μοντέλου
    infer_model = load_model(config['train']['saved_weights_name'])

    # Υπολογισμός mAP για όλες τις κλάσεις
    iou_threshold =  config["valid"]["iou_thresh"] if config["valid"]["iou_thresh"] else 0.5
    average_precisions = evaluate(infer_model, valid_generator, iou_threshold=iou_threshold)

    # Εκτύπωση αποτελέσματος
    print(f"AP and mAP at {iou_threshold} iou_threshold")
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file', required=True)    
    
    args = argparser.parse_args()
    _main_(args)
