# import the necessary modules
from utils import utils_pose as up
import cv2

if __name__=='__main__':
    # Initialize detector and estimator
    detector = up.load_detector()
    #------------------------
    estimator_name = 'simple_pose_resnet50_v1b'

    HASHTAG = True
    output_shape = (256, 192)
    
    estimator = up.load_pose_estimator(estimator_name, hashtag=HASHTAG)

    video_path = '/home/cuong/Desktop/Speaking/GreenGlobal/Tram_Trai_30.mp4'
    output_path = 'outputs/video_1_{}.mp4'.format(estimator_name)
    time_stamp = 1
    wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
    up.visualize_poses_in_a_video(video_path, detector, estimator, output_shape, 
                                wanted_joints, scale=0.5, output_path=output_path, time_stamp=time_stamp)
    

