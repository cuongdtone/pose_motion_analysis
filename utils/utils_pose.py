# import the necessary modules
from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import torch
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
#from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

from .utils_viz import cv_plot_image, cv_plot_keypoints, cv_plot_upper_keypoints
from .utils_transforms import transform_test

#KEYPOINTS = data.mscoco.keypoints.COCOKeyPoints.KEYPOINTS
KEYPOINTS = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}

def load_detector(detector_name="ssd_512_mobilenet1.0_coco", ctx=None):
    if ctx is None:
        ctx = mx.cpu()
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    ## To speed up the detector, 
    ## we can reset the prediction head to only include the classes we need.
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    detector.hybridize()
    return detector
    
def load_pose_estimator(estimator_name='simple_pose_resnet18_v1b', hashtag=True, ctx=None):
    ## The default simple_pose_resnet18_v1b model 
    ## was trained with input size 256x192.
    ## We also provide an optional simple_pose_resnet18_v1b model
    ## trained with input size 128x96.
    if ctx is None:
        ctx = mx.cpu()
    if not hashtag:
        hashtag = 'ccd24037'
    estimators = get_model(estimator_name, pretrained=hashtag, ctx=ctx)
    estimators.hybridize()
    return estimators

def get_final_detections(class_IDs, scores, bounding_boxs, threshold=0.2):
    L = class_IDs.shape[1]
    new_class_IDs = []
    new_scores = []
    new_bounding_boxs = []
    ids = []
    for i in range(L):
        if class_IDs[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < threshold:
            continue
        ids.append(i)
        new_class_IDs.append(class_IDs[0][i].asnumpy().tolist())
        new_scores.append(scores[0][i].asnumpy().tolist())
        new_bounding_boxs.append(bounding_boxs[0][i].asnumpy().tolist())
    
    new_class_IDs = mx.nd.array(new_class_IDs)
    new_scores = mx.nd.array(new_scores)
    new_bounding_boxs = mx.nd.array(new_bounding_boxs)
    
    if len(ids) > 0:
        new_class_IDs = new_class_IDs.reshape((1, len(ids), 1))
        new_scores = new_scores.reshape((1, len(ids), 1))
        new_bounding_boxs = new_bounding_boxs.reshape((1, len(ids), 4))
    return new_class_IDs, new_scores, new_bounding_boxs


def get_wanted_joints(pred_coords, confidence, wanted_joints=None):
    if wanted_joints is None:
        return pred_coords, confidence
    
    global KEYPOINTS
    
    # Create new data
    new_coords = np.zeros((pred_coords.shape[0], len(wanted_joints), 2))
    new_confidence = np.zeros((confidence.shape[0], len(wanted_joints), 1))
    
    # Select keypoints which are not wanted
    keypoint_ids = []
    for keypoint_id in range(len(KEYPOINTS)):
        keypoint = KEYPOINTS[keypoint_id]
        if keypoint not in wanted_joints:
            keypoint_ids.append(keypoint_id)

    # Delete the selected keypoints
    for cid, (coord, conf) in enumerate(zip(pred_coords, confidence)):
        coord = coord.asnumpy()
        conf = conf.asnumpy()
        new_coords[cid] = np.delete(coord, keypoint_ids, axis=0)
        new_confidence[cid] = np.delete(conf, keypoint_ids, axis=0)
    new_coords = mx.nd.array(new_coords)
    new_confidence = mx.nd.array(new_confidence)    

    return new_coords, new_confidence


def get_estimations(image_bgr, detector, estimator, output_shape=(128,96), ctx=None, wanted_joints=None):
    if ctx is None:
        ctx = mx.cpu()
    image_mx = mx.nd.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).astype('uint8')
    # Transform the image to the input size
    x, image_mx = transform_test(image_mx, short=512, max_size=350)
    x = x.as_in_context(ctx)
    
    # Predict locations of people
    class_IDs, scores, bounding_boxs = detector(x)
    
    # Filter predictions using a threshold
#    print('#### Scores: ', scores)
    class_IDs, scores, bounding_boxs = get_final_detections(class_IDs, scores, bounding_boxs, threshold=0.3)

    # Estimate human poses corresponding to 'bounding_boxs'
    pose_input, upscale_bbox = detector_to_simple_pose(image_mx, class_IDs, scores, bounding_boxs, output_shape=output_shape, ctx=ctx)
    
    # Get corresponding features (a list of tensors(x_i): [x_1, y_1, x_2, y_2, ...])
    bboxes_output = []
    poses_output = []
    features_output = []
    pred_coords = []
    confidence = []
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        pred_coords, confidence =  get_wanted_joints(pred_coords, confidence, wanted_joints)
    
    # Output
    data = {}
    data['image_mx'] = [image_mx]
    data['detections'] = [class_IDs, scores, bounding_boxs]
    data['estimations'] = [pred_coords, confidence]
    return data


def get_features(pred_coords):
    features_list = []
    if len(pred_coords) > 0:
        for pred_coord in pred_coords:
            p = pred_coord.asnumpy()
            features = []
            for i, row in enumerate(p):
                features += [torch.tensor(x, dtype=torch.float32) for x in row]
            features_list.append(features)
    return features_list


def rescale_a_point(point, original_size=(1280,720), scaled_size=(350,197)):
    w_ratio = original_size[0]/scaled_size[0]
    h_ratio = original_size[1]/scaled_size[1]
    new_x = point[0] * w_ratio;
    new_y = point[1] * h_ratio;
    new_point = [new_x, new_y]
    return new_point


def rescale_points(points, original_size=(1280,720), scaled_size=(350,197)):
    new_points = []
    for point in points:
        new_point = rescale_a_point(point, original_size, scaled_size)
        new_points.append(new_point)
    return new_points


def rescale_coords(coords, original_size=(1280,720), scaled_size=(350,197)):
    data = np.zeros(coords.shape)
    for i in range(coords.shape[0]):
        pts = coords[i]
        for j in range(pts.shape[0]):
            point = pts[j].squeeze().asnumpy().tolist()
#            print('point: ', point)
            new_point = rescale_a_point(point, original_size, scaled_size)
#            print('new_point: ', new_point)
            data[i, j][0] = new_point[0]
            data[i, j][1] = new_point[1]
    return data
   
    
def rescale_bboxes(bboxes, original_size=(1280,720), scaled_size=(350,197)):
    new_bboxes = np.zeros(bboxes.shape)
    for i in range(bboxes.shape[0]):
        for j in range(bboxes.shape[1]):
            bb = bboxes[i,j].squeeze().asnumpy().tolist()
            point_1 = [bb[0], bb[1]]
            point_2 = [bb[2], bb[3]]
            points = [point_1, point_2]
            new_points = rescale_points(points, original_size, scaled_size)
            new_bboxes[i,j][0] = new_points[0][0]
            new_bboxes[i,j][1] = new_points[0][1]
            new_bboxes[i,j][2] = new_points[1][0]
            new_bboxes[i,j][3] = new_points[1][1]
    return new_bboxes


def extract_poses_from_frames(frames, detector, estimator, output_shape=(128,96), num_joints=17):
    '''
    Args:
        frames: A list of video frames
    Return:
        numpy.ndarray of (num_frames, num_joints=17, 2)
    '''
    poses = np.zeros((len(frames), num_joints, 2))
    for i in range(len(frames)):
        frame = frames[i]
        data = get_estimations(frame, detector, estimator, output_shape)
        pred_coords, confidence = data['estimations']
        if len(pred_coords) == 1:
            for pred_coord in pred_coords:
                p = pred_coord.asnumpy()
                for j, row in enumerate(p):
                    poses[i,j][0] = row[0]
                    poses[i,j][1] = row[1]
    return poses


def extract_poses_from_a_video(video_path, detector, estimator, output_shape=(128,96), num_joints=17):
    '''
    Args:
        video_path: A path of a video
    Return:
        numpy.ndarray of (num_frames, num_joints=17, 2)
    '''
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)  ### letting the camera autofocus
    frames = []
    poses = None
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames)>=32:
            poses = extract_poses_from_frames(frames, detector, estimator, output_shape, num_joints)
            frames = []
            break
    return poses


def extract_poses_to_text(video_path, detector, estimator, output_shape=(128,96), txt_path='./joints_output.txt'):
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)  ### letting the camera autofocus
    fid = open(txt_path, 'w')
    ## -------------
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        data = get_estimations(frame, detector, estimator, output_shape)
#        image_mx = data['image_mx'][0]
#        class_IDs, scores, bounding_boxs = data['detections']
        pred_coords, confidence = data['estimations']
        
        # Extract to a text file
        if len(pred_coords) > 0:
            for pred_coord in pred_coords:
                p = pred_coord.asnumpy()
                coords_str = '' # j0_x,  j0_y, j1_x, j1_y , ... , j17_x, j17_y
                for i, row in enumerate(p):
                    coords_str += '{:.3f},{:.3f},'.format(row[0], row[1])
                fid.write(coords_str[:-1]+'\n')
    ## -------------
    fid.close()
    cap.release()
    
from utils.ultis_cal import *
from filter import butter_lowpass_filter
def visualize_poses_in_a_video(video_path, detector, estimator, output_shape=(128,96), 
                            wanted_joints=None, scale=1.0, output_path=None, time_stamp=1):
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)  ### letting the camera autofocus
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_dim = (new_width, new_height)
    if output_path is not None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    else:
        num_frames = 2
    #inittial analysic
    mat_feature = []
    frames_analysic = 64
    
    #-------------------------------------------------
    # Estimation loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_shape = frame.shape
        original_size = (frame_shape[1], frame_shape[0])
        data = get_estimations(frame, detector, estimator, output_shape, wanted_joints=wanted_joints)
        image_mx = data['image_mx'][0]
        img_mx_shape = image_mx.shape
        scaled_size = (img_mx_shape[1], img_mx_shape[0])
        class_IDs, scores, bounding_boxs = data['detections']
#        print('scores: ', scores)
#        print('bboxes: ', bounding_boxs.shape)
        pred_coords, confidence = data['estimations']
#        print(pred_coords)
        if len(pred_coords)>0:
            lanmarks = pred_coords[0].asnumpy()
            #print(lanmarks)
            cv2.imshow(' landmark before norm', visualize_landmark_before_norm(lanmarks))
            lanmarks = normalize_pose_landmarks(lanmarks)

            feature = feature_etractor(lanmarks)
            mat_feature.append(feature)
        if len(mat_feature) >= frames_analysic:
            mat_feature = np.array(mat_feature)
            figure = plt.figure()
            standard_devication = []
            for f in range(0, 4):
                alpha1 = mat_feature[:, f]
                #rint(alpha1)
                alpha = butter_lowpass_filter(alpha1)
                #alpha1 = alpha1[8:] #ignore first noise
                standard_devication.append(np.std(alpha))
                plt.subplot(2,2,f+1)
                plt.axis([0, frames_analysic, -50, 50])
                plt.title(feature_name[f])
                plt.xlabel('frames')
                plt.ylabel('value (%)')
                plt.plot(range(len(alpha)), alpha, 'b')
            out = np.mean(standard_devication)
            print('Thress hold: ', out)
            # predict
            if out>2:
                print('Dang trinh dien')
            else:
                print('Khong trinh dien')
            figure.savefig(video_path.split('.')[0]+'.png', dpi=250)
            mat_feature = []
        plt.show()
        plt.pause(0.05)
        # Visualization
        out_img = frame.copy()
        if len(pred_coords)>0:
            frame_mx = mx.nd.array(frame).astype('uint8')
            new_bboxes = rescale_bboxes(bounding_boxs, original_size, scaled_size)
            new_coords = rescale_coords(pred_coords, original_size, scaled_size)
            img = cv_plot_upper_keypoints(frame_mx, new_coords, confidence, class_IDs, new_bboxes, scores, box_thresh=0.5, keypoint_thresh=0.2)
#            img = cv_plot_keypoints(image_mx, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)
            
            # Features
#            features_list = get_features(pred_coords)
    #        print('features_list: ', features_list)
    #        print('bounding_boxs: ', bounding_boxs)
            
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#            print('Output shape: ', img.shape)
            # resize image
            resized = cv2.resize(out_img, new_dim, interpolation = cv2.INTER_AREA)
            cv_plot_image(resized)
            
#            visualize_poses_in_an_image(frame, pred_coords, confidence)
        # Write to a video file
        if output_path is not None:
            resized = cv2.resize(out_img, new_dim, interpolation = cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            video_writer.write(resized)
        key = cv2.waitKey(time_stamp)
        if key==ord('q'):
            break

    cap.release()
    if output_path is not None:
        video_writer.release()


def visualize_poses_in_an_image(image_bgr, pred_coords, confidence, scaled_size=(350,197)):
    from PIL import Image, ImageDraw
    pil_image = Image.fromarray(image_bgr)
    draw = ImageDraw.Draw(pil_image)
    
    h, w = image_bgr.shape[:2]
    original_size = (w,h)
    
    keypoints = data.mscoco.keypoints.COCOKeyPoints.KEYPOINTS
    
    for keypoint_id in range(len(keypoints)):
        print('{} : {}'.format(keypoint_id, keypoints[keypoint_id]))
        pred = pred_coords[:,keypoint_id,:]
        for i in range(pred.shape[0]):
            if (confidence[i,keypoint_id,:] > 0.2) == 1:
                pos = pred[i,:].asnumpy()
                new_pos = rescale_a_point(pos, original_size, scaled_size)
                draw.text(new_pos, text=keypoints[keypoint_id], fill='red')
    pil_image.show()


def try_pose_estimation_in_an_image(image_bgr, detector, estimator, output_shape=(128,96), wanted_joints=None, visualize=True):
    image_shape = image_bgr.shape
    original_size = (image_shape[1], image_shape[0])
    # Get human poses
    data = get_estimations(image_bgr, detector, estimator, output_shape, wanted_joints=wanted_joints)
    image_mx = data['image_mx'][0]
    img_mx_shape = image_mx.shape
    scaled_size = (img_mx_shape[1], img_mx_shape[0])
    class_IDs, scores, bounding_boxs = data['detections']
    pred_coords, confidence = data['estimations']
    
    # Visualization
    out_img = image_bgr.copy()
    if len(pred_coords)>0:        
        image_mx = mx.nd.array(image_bgr).astype('uint8')
        new_bboxes = rescale_bboxes(bounding_boxs, original_size, scaled_size)
        new_coords = rescale_coords(pred_coords, original_size, scaled_size)
        img = cv_plot_upper_keypoints(image_mx, new_coords, confidence, class_IDs, new_bboxes, scores, box_thresh=0.5, keypoint_thresh=0.2)
        if visualize:
            cv_plot_image(img)
            cv2.waitKey(0)
        out_img = img.copy()
    return out_img


def perform_measure(video_path, detector, estimator, output_shape=(128, 96),
                               wanted_joints=None, scale=1.0, output_path=None, time_stamp=1, step_frame = 1):
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)  ### letting the camera autofocus

    width = int(cap.get(3))
    height = int(cap.get(4))
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_dim = (new_width, new_height)
    if output_path is not None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    else:
        num_frames = 2
    # inittial analysic
    mat_feature = []
    frames_analysic = 64
    last_time = 0
    last_frame = -step_frame
    # -------------------------------------------------
    # Estimation loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if not last_frame + step_frame == i:
            continue
        last_frame += step_frame
        if last_time!=0:
            period = time.time() - last_time
            FPS = 1/period
            print("FPS = ", FPS)
        last_time = time.time()
        frame_shape = frame.shape
        original_size = (frame_shape[1], frame_shape[0])
        data = get_estimations(frame, detector, estimator, output_shape, wanted_joints=wanted_joints)
        image_mx = data['image_mx'][0]
        img_mx_shape = image_mx.shape
        scaled_size = (img_mx_shape[1], img_mx_shape[0])
        class_IDs, scores, bounding_boxs = data['detections']
        #        print('scores: ', scores)
        #        print('bboxes: ', bounding_boxs.shape)
        pred_coords, confidence = data['estimations']
        #        print(pred_coords)
        if len(pred_coords) > 0:
            lanmarks = pred_coords[0].asnumpy()
            # print(lanmarks)
            cv2.imshow(' landmark before norm', visualize_landmark_before_norm(lanmarks))
            lanmarks = normalize_pose_landmarks(lanmarks)

            feature = feature_etractor(lanmarks)
            mat_feature.append(feature)
        if len(mat_feature) >= frames_analysic:
            mat_feature = np.array(mat_feature)
            figure = plt.figure()
            standard_devication = []
            for f in range(0, 4):
                alpha1 = mat_feature[:, f]
                # rint(alpha1)
                alpha = butter_lowpass_filter(alpha1)
                # alpha1 = alpha1[8:] #ignore first noise
                standard_devication.append(np.std(alpha))
                plt.subplot(2, 2, f + 1)
                plt.axis([0, frames_analysic, -50, 50])
                plt.title(feature_name[f])
                plt.xlabel('frames')
                plt.ylabel('value (%)')
                plt.plot(range(len(alpha)), alpha, 'b')
            out = np.mean(standard_devication)
            print('Thress hold: ', out)
            # predict
            if out > 2:
                print('Dang trinh dien')
            else:
                print('Khong trinh dien')
            figure.savefig(video_path.split('.')[0] + '.png', dpi=250)
            mat_feature = []
        plt.show()
        plt.pause(0.05)
        # Visualization
        out_img = frame.copy()
        if len(pred_coords) > 0:
            frame_mx = mx.nd.array(frame).astype('uint8')
            new_bboxes = rescale_bboxes(bounding_boxs, original_size, scaled_size)
            new_coords = rescale_coords(pred_coords, original_size, scaled_size)
            img = cv_plot_upper_keypoints(frame_mx, new_coords, confidence, class_IDs, new_bboxes, scores,
                                          box_thresh=0.5, keypoint_thresh=0.2)
            #            img = cv_plot_keypoints(image_mx, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)

            # Features
            #            features_list = get_features(pred_coords)
            #        print('features_list: ', features_list)
            #        print('bounding_boxs: ', bounding_boxs)

            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #            print('Output shape: ', img.shape)
            # resize image
            resized = cv2.resize(out_img, new_dim, interpolation=cv2.INTER_AREA)
            cv_plot_image(resized)

        #            visualize_poses_in_an_image(frame, pred_coords, confidence)
        # Write to a video file
        if output_path is not None:
            resized = cv2.resize(out_img, new_dim, interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            video_writer.write(resized)
        key = cv2.waitKey(time_stamp)
        if key == ord('q'):
            break

    cap.release()
    if output_path is not None:
        video_writer.release()


if __name__=='__main__':

    video_path = '/media/hueailab/HDPC-UT/GreenGlobal-AI/Projects/DLTM/Code/HanhVi/Ref/HAR_Pose_LSTM/video1_clip1.mp4'
    txt_path = './joints_output_1.txt'

    detector = load_detector()
    estimator = load_pose_estimator()
    
#    extract_poses_to_text(video_path, detector, estimator, txt_path)
    
    image_path = 'samples/sample_01.png'
    image_bgr = cv2.imread(image_path)
    
#    cv2.imshow('Image', image_bgr)
#    cv2.waitKey(0)
    
    wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
    out_img = try_pose_estimation_in_an_image(image_bgr, detector, estimator, output_shape=(128,96), wanted_joints=wanted_joints)
    
    keypoints = data.mscoco.keypoints.COCOKeyPoints.KEYPOINTS
    print('[INFO] Keypoints: ', keypoints)  # {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}

