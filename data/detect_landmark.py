# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames


def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_fn):
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
    input_dir = root_dir #
    output_dir = landmark_dir #
    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    fids.sort()

    print(f"{len(fids)} files")
    for fid in tqdm(fids):
        output_fn = os.path.join(output_dir, fid+'.pkl')
        video_path = os.path.join(input_dir, fid+'.mp4')
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return


if __name__ == '__main__':
    import argparse
    import skvideo
    import skvideo.io

    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--landmark', type=str, help='landmark dir')
    parser.add_argument('--manifest', type=str, help='a list of filenames')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    parser.add_argument('--face_preprocess_dir', type=str, help='ffmpeg path')
    args = parser.parse_args()
    model_path=args.face_preprocess_dir
    face_predictor_path = os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
    cnn_detector_path = os.path.join(model_path, "mmod_human_face_detector.dat")

    vid_dir = os.path.join(args.root)
    detect_face_landmarks(face_predictor_path, cnn_detector_path, vid_dir, args.landmark, args.manifest)
