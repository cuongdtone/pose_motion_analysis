import numpy as np
import cv2

def draw_point(image, points):
    for i in points:

        cv2.circle(image, (int(i[0]), int(i[1])), 3, (0, 0, 255), 1)

def visualize_landmark_before_norm(landmark):

    image = np.zeros([197, 350, 3])
    draw_point(image, landmark)
    return image

def get_pose_center(left_hip, right_hip):
    return (left_hip+right_hip)*0.5
def get_pose_size(landmarks):
    nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = landmarks
    hips = (left_hip + right_hip) * 0.5
    shoulders = (left_shoulder + right_shoulder) * 0.5
    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)
    #print(shoulders, hips)
    #print(torso_size)
    # Max dist to pose center.
    pose_center = get_pose_center(left_hip, right_hip)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))
    return max(torso_size * 2.5, max_dist)
def normalize_pose_landmarks(landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)
    nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = landmarks
    # Normalize translation.
    pose_center = get_pose_center(left_hip, right_hip)
    landmarks -= pose_center
    # Normalize scale.
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100
    return landmarks
def cal_angle_three_point(a, b, c):
    #a = np.array([-1, 0])
    #b = np.array([0, 0])
    #c = np.array([1, 0])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
def euclid(a, b):
    return np.linalg.norm(a-b)
def feature_etractor(pred_coords):
    if len(pred_coords)>0:
        nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = pred_coords
        #left_elbow_angle = cal_angle_three_point(left_wrist, left_elbow, left_shoulder)/180
        #right_elbow_angle = cal_angle_three_point(right_wrist, right_elbow, right_shoulder)/180
        #right_arm_hip_d = euclid(right_wrist, right_hip)
        #left_arm_hip_d = euclid(left_wrist, left_hip)
        #left_right_hand = euclid(left_wrist, right_wrist)
        #right_arm_d = euclid(right_wrist, right_shoulder)
        #left_arm_d = euclid(left_wrist, left_shoulder)
        #pose_center = get_pose_center(left_hip, right_hip)
        #angle = cal_angle_three_point(left_wrist, pose_center, right_wrist)
        feature = [left_wrist[0], left_wrist[1], right_wrist[0], right_wrist[1]]
        return feature

feature_name = ['x', 'y', 'x', 'y']