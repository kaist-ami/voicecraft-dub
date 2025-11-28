#/saltpool0/data/sungbin/CVPR24/lip2speech/1011_sanity/original/train/anger
import os
import glob
import pdb
import shutil
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from tqdm import tqdm
import json
save_dir = "/saltpool0/data/sungbin/CVPR24/lip2speech/1017_sanity/original/val"
source_dir = "/saltpool0/data/sungbin/CVPR24/dataset/celebvtext/celebvtext_split_music/video"
with open("/saltpool0/data/sungbin/CVPR24/dataset/celebvtext/celebvtext_split_music/emotion_dic.json") as f:
    json_file = json.load(f)

mp4_list = sorted(glob.glob(os.path.join(source_dir, "*")))[44000:]

for mp4 in tqdm(mp4_list):
    try:
        mp4_name = mp4.split("/")[-1][:-6]
        emotion = json_file[mp4_name]
    except:
        print("except")
        try:
            mp4_name = mp4.split("/")[-1][:-7]
            emotion = json_file[mp4_name]
        except:
            mp4_name = mp4.split("/")[-1][:-6]
            emotion = "neutral"

    save_emotion = os.path.join(save_dir, emotion)
    if not os.path.isdir(save_emotion):
        os.makedirs(save_emotion, exist_ok=True)
    shutil.copy(mp4, os.path.join(save_emotion, mp4.split("/")[-1]))

