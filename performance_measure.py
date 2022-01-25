from utils import utils_pose as up
import os

total_memory, used_memory_before, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("Total memory: ", total_memory)

detector = up.load_detector()
# ------------------------
estimator_name = 'simple_pose_resnet101_v1b'
step_frame = 2
frame_analysis = 64

HASHTAG = True
output_shape = (256, 192)

estimator = up.load_pose_estimator(estimator_name, hashtag=HASHTAG)

video_path = 'samples/video1_clip1.mp4'
output_path = 'outputs/video_1_{}.mp4'.format(estimator_name)
time_stamp = 1

wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                 'left_hip', 'right_hip']

total_memory, used_memory_after, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("Used memmory for model: ", used_memory_after-used_memory_before, "MB")
up.perform_measure(video_path, detector,
                   estimator, output_shape,
                   wanted_joints,
                   scale=0.5,
                   output_path=output_path,
                   time_stamp=time_stamp, step_frame=step_frame, frames_analysic=frame_analysis)
