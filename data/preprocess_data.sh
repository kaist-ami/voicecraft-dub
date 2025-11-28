#!/bin/bash

ENCODEC=../pretrained_models/encodec.th
SPLIT_NAME=trainval
ROOT_DIR_PATH=../samples
SAVE_DIR_PATH="../samples/${SPLIT_NAME}_preprocess"
FACE_PREPROCESS=../pretrained_models/landmarks

#Extract wav file from video and save file.list
python save_wav.py \
    --root_dir "$ROOT_DIR_PATH" \
    --split_name "$SPLIT_NAME"

## Extract audio tokens and phoneme tokens
python phonemize_lrs.py \
    --save_dir "$SAVE_DIR_PATH" \
    --root_dir "$ROOT_DIR_PATH" \
    --encodec_model_path "$ENCODEC" \
    --split_name "$SPLIT_NAME"

## Preprocess lip video
python detect_landmark.py --root ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark \
 --manifest ${ROOT_DIR_PATH}/file.list --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}

python align_mouth.py --video-direc ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark --filename-path ${ROOT_DIR_PATH}/file.list \
 --save-direc ${SAVE_DIR_PATH}/video  --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}

#Make a continuation form for constructing training dataset
python construct_dataset.py --root_dir ${SAVE_DIR_PATH}
