# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys,math
import cv2,dlib,time
import os,pickle,shutil,tempfile
import math
import pdb
import cv2
import glob
import subprocess
import numpy as np
from collections import deque
from skimage import transform as tf
from tqdm import tqdm
import argparse
import skvideo
import skvideo.io
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals) + '.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir + '/%0' + str(decimals) + 'd.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20',
           target_path]
    pipe = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return

def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx,
               stop_idx, crop_height, crop_width):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    print(video_pathname)
    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)

    cnt = 0

    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break

        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)

        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame,
                                      trans_landmarks[start_idx:stop_idx],
                                      crop_height // 2,
                                      crop_width // 2, ))
        # if frame_idx == len(landmarks)-1:
        if frame_idx == len(landmarks) - 2:
            q_landmarks.append(landmarks[frame_idx])
            q_frame.append(frame)

            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame,
                                          trans_landmarks[start_idx:stop_idx],
                                          crop_height // 2,
                                          crop_width // 2, ))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_face_landmarks(file_path, model_path):
    face_predictor_path=os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
    cnn_detector_path = os.path.join(model_path, "mmod_human_face_detector.dat")

    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)

    output_fn = file_path.replace(".mp4",".pkl")
    video_path = file_path
    frames = load_video(video_path)
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, cnn_detector, predictor)
        landmarks.append(landmark)
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    pickle.dump(landmarks, open(output_fn, 'wb'))

def align_mouth(file_path, model_path):
    crop_width= 96
    crop_height=96
    ffmpeg="ffmpeg"
    window_margin=12
    start_idx=48
    stop_idx=68

    # -- mean face utils
    STD_SIZE = (256, 256)
    mean_face_path = os.path.join(model_path, "20words_mean_face.npy")
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    video_pathname = file_path

    landmarks_pathname = file_path.replace(".mp4",".pkl")
    dst_pathname = file_path.replace(".mp4","_lip.mp4")

    assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
    assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)

    landmarks = pickle.load(open(landmarks_pathname, 'rb'))

    # -- pre-process landmarks: interpolate frames not being detected.
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if not preprocessed_landmarks:
        print(f"resizing {filename}")
        frame_gen = read_video(video_pathname)
        frames = [cv2.resize(x, (crop_width, crop_height)) for x in frame_gen]
        write_video_ffmpeg(frames, dst_pathname, ffmpeg)

    # -- crop
    sequence = crop_patch(video_pathname, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                          window_margin=window_margin, start_idx=start_idx, stop_idx=stop_idx,
                          crop_height=crop_height, crop_width=crop_width)
    # assert sequence is not None, "cannot crop from {}.".format(filename)

    # -- save
    os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
    write_video_ffmpeg(sequence, dst_pathname, ffmpeg)
    return dst_pathname
