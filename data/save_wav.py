# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys, glob, subprocess, json, math
import pdb
import librosa, cv2
import numpy as np
from scipy.io import wavfile
from os.path import basename, dirname
from tqdm import tqdm
import tempfile, shutil
import soundfile as sf
import os
import multiprocessing
import argparse
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def get_filelist(root_dir, split):
    fids = []
    all_fns = glob.glob(f"{root_dir}/{split}/*/*.mp4")
    for fn in all_fns:
        fids.append('/'.join(fn.split('/')[-3:])[:-4])

    output_fn = f"{root_dir}/file.list"
    with open(output_fn, 'w') as fo:
        fo.write('\n'.join(fids)+'\n')
    return

def prep_wav(root_dir, flist, ffmpeg):
    input_dir = root_dir
    fids = [ln.strip() for ln in open(flist).readlines()]
    fids.sort()
    print(f"{len(fids)} videos")
    for i, fid in enumerate(tqdm(fids)):
        video_fn = f"{input_dir}/{fid}.mp4"
        audio_fn_ = f"{input_dir}/{fid}_.wav"
        audio_fn = f"{input_dir}/{fid}.wav"
        if os.path.isfile(audio_fn): continue
        try:
            os.makedirs(os.path.dirname(audio_fn), exist_ok=True)
            cmd = ffmpeg + " -i " + video_fn + " -f wav -vn -y " + audio_fn_ + ' -loglevel quiet'
            subprocess.call(cmd, shell=True)

            #ffmpeg -i input.wav -ar 16000 output.wav
            cap = cv2.VideoCapture(video_fn)
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            aud_f, sr = librosa.load(audio_fn_, sr=16000)
            aud_f = aud_f[:int(num_frames_video/25*16000)]
            sf.write(audio_fn, aud_f, 16000)
            os.remove(audio_fn_)
        except:
            pdb.set_trace()
            continue
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save_audio', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--split_name', type=str)
    parser.add_argument('--ffmpeg', type=str, default="ffmpeg", help='ffmpeg path')
    parser.add_argument('--step', type=int, help='Steps(1: get file list, 2: extract audio)')
    args = parser.parse_args()

    print(f"Get file list")
    get_filelist(args.root_dir, args.split_name)

    print(f"Extract audio")
    manifest = f"{args.root_dir}/file.list"
    prep_wav(args.root_dir, manifest, args.ffmpeg)

