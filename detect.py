#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    # os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # run on CPU

    infer_model_ts = load_model(config['ts_model'])
    infer_model_tf = load_model(config['tf_model'])


    ###############################
    #   Predict bounding boxes 
    ############################### 
    if input_path[-4:] == '.avi': # do detection on a video  
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'DIVX'), 
                               cv2.CAP_PROP_FRAME_COUNT, 
                               (frame_w, frame_h))
        # the main loop
        images      = []
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            images += [image]

            # predict the bounding boxes
            batch_boxes_ts = get_yolo_boxes(infer_model_ts, images, net_h, net_w, config['anchors'], obj_thresh, nms_thresh)
            batch_boxes_tf = get_yolo_boxes(infer_model_tf, images, net_h, net_w, config['anchors'], obj_thresh, nms_thresh)

            for i in range(len(images)):
                # draw bounding boxes on the image using labels
                draw_boxes(images[i], batch_boxes_ts[i], config['ts_labels'], obj_thresh)   
                draw_boxes(images[i], batch_boxes_tf[i], config['tf_labels'], obj_thresh)   

                # write result to the output video
                video_writer.write(images[i]) 
            images = []
            if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model_ts, [image], net_h, net_w, config['anchors'], obj_thresh, nms_thresh)[0]
            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['ts_labels'], obj_thresh) 

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model_tf, [image], net_h, net_w, config['anchors'], obj_thresh, nms_thresh)[0]
            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['tf_labels'], obj_thresh)

            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('\\')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
