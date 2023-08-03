"""
Most of the code to extract the Region of Interest is based on Pingchuan Ma's implementation:
https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/

"""
import os
import cv2
import pickle
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def load_video(video_path):
    """load a video.

    Args:
        video_path (str): the path where the video file is stored.
    """
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def get_video_info(video_path):
    """get video information.

    Args:
        video_path (str): the path where the video file is stored.

    Returns:
        tuple: (fps, frame_count, frame_width, frame_height)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, frame_width, frame_height

def landmarks_interpolate(landmarks):
    """landmarks interpolation.

    Args:
        landmarks (numpy.ndarray): the facial landmarks for each video frame.

    Returns:
        numpy.ndarray: the facial landmarks for each video frame interpolated.
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    Args:
        landmarks (numpy.ndarray): input facial landmarks to be interpolated.
        start_idx (int): the start index for linear interpolation.
        stop_idx (int): the stop for linear interpolation.

    Returns:
        numpy.ndarray: facial landmarks linearly interpolated.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def affine_transform(
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0
    ):
        """applies the affine transform.

        Args:
            frame (numpy.ndarray): the input video frame.
            landmarks (numpy.ndarray): the tracked facial landmarks.
            reference (numpy.ndarray): the neutral reference video frame.
            grayscale (bool): convert the video frame image in grayscale if set as True.
            target_size (tuple): size of the video frame output image.
            reference_size (tuple): size of the neutral reference video frame.
            stable_points (tuple): landmark indices for the stable points.
            interpolation: interpolation method to be used.
            border_mode: pixel extrapolation method.
            border_value: value used in case of a constant border. By default, it is 0.

        Returns:
            numpy.ndarray: the transformed video frame
            numpy.ndarray: the transformed facial landmarks
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks

def cut_patch(frame, landmarks, height, width, threshold=5):
    """cutting the facial patch.

    Args:
        frame (numpy.ndarray): the input video frame.
        landmarks (numpy.ndarray), the corresponding facial landmarks for the input video frame.
        height (int): the distance from the centre to the upper/bottom side of of a bounding box.
        width (int): the distance from the centre to the left/right side of of a bounding box.
        threshold (int): the threshold from the centre of a bounding box to the side of image.

    Returns:
        numpy.ndarray: the face patch.
    """
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > frame.shape[0]:
        center_y = frame.shape[0] - height
    if center_y + height > frame.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > frame.shape[1]:
        center_x = frame.shape[1] - width
    if center_x + width > frame.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_frame = np.copy(frame[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_frame

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  """
    Code provided by MediaPipe to draw the landmarks on a detect person,
  as well as the expected connections between those markers.
  """

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

  return annotated_image
