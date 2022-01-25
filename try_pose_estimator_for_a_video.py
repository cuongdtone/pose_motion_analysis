# import the necessary modules
from utils import utils_pose as up
import cv2

if __name__=='__main__':
    # Initialize detector and estimator
    detector = up.load_detector()
    
    #------------------------
    estimator_name = 'simple_pose_resnet50_v1b'
#    estimator_name = 'mobile_pose_resnet50_v1b'
    HASHTAG = True
    output_shape = (256,192)
    #------------------------
#    estimator_name = 'simple_pose_resnet18_v1b' # this model returns outputs of shape by (128,96)
#    HASHTAG = False
#    output_shape = (128,96)
    
    estimator = up.load_pose_estimator(estimator_name, hashtag=HASHTAG)
    
    
    #--------------------------------
    # Visualize human poses in a video
    #--------------------------------
    video_path = 'An extra Eye/An extra Eye An extra Ear An extra Heart_clip_00000028.mp4'
    output_path = 'outputs/video_1_{}.mp4'.format(estimator_name)
#    output_path = None
    time_stamp = 1
    
    wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
    
    up.visualize_poses_in_a_video(video_path, detector, estimator, output_shape, 
                                wanted_joints, scale=0.5, output_path=output_path, time_stamp=time_stamp)
    

