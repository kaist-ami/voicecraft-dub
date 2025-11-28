#!/bin/bash

SOURCE_TEXT_PATH=./samples/test/eZj5n8ScTkI/00001.txt
SOURCE_AUDIO_PATH=./samples/test/eZj5n8ScTkI/00001.wav
TARGET_TEXT_PATH=./samples/test/eZj5n8ScTkI/00005.txt
TARGET_VIDEO_PATH=./samples/test/eZj5n8ScTkI/00005.mp4
VOICECRAFT_DUB=./pretrained_models/voicecraft_dub
ENCODEC=./pretrained_models/encodec.th
FACE_PREPROCESS=./pretrained_models/landmarks
SAVE_DIR_PATH=./results

python inference_dubbing.py \
    --model_dir "$VOICECRAFT_DUB" \
    --encodec_dir "$ENCODEC" \
    --face_preprocess_dir "$FACE_PREPROCESS" \
    --result_dir "$SAVE_DIR_PATH" \
    --src_text "$SOURCE_TEXT_PATH" \
    --src_audio "$SOURCE_AUDIO_PATH" \
    --tar_text "$TARGET_TEXT_PATH" \
    --tar_vid "$TARGET_VIDEO_PATH" \
    --sample_batch_size 3