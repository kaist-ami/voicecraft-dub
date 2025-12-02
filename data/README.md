# VoiceCraft-Dub Data Preprocessing
This directory contains scripts for data preparation (LRS3 or CelebV-Dub) for training VoiceCraft-Dub.

## Download source dataset
Download [LRS3](https://mmai.io/datasets/lip_reading/) and [CelebV-Dub](https://drive.google.com/file/d/1pL4C4sRiQimbsFHXQKYx6NNyjOhPvR3l/view?usp=sharing).

## Run preprocessing
```
sh preprocess_data.sh
```

## Details
### Arguments
Note that the hierarchy of the directory should follow the example given in ../samples, such as the split name, and the videos are categorized with the same speaker.

```
ENCODEC=../pretrained_models/encodec.th #pretrained encodec directory
SPLIT_NAME=trainval #split name of each dataset
ROOT_DIR_PATH=../samples #in samples, there is the split named trainval
SAVE_DIR_PATH="../samples/${SPLIT_NAME}_preprocess" #directory to save the preprocessed results
FACE_PREPROCESS=../pretrained_models/landmarks #pretrained models for lip processsing
```

### Extract wav files
Given the video, the wav files are extracted and the metadata for trainnig is saved in file.list.
```
python save_wav.py \
    --root_dir "$ROOT_DIR_PATH" \
    --split_name "$SPLIT_NAME"
```


### Extract audio tokens and phoneme tokens
Given the wav file and the txt file containing the transcript, tokens are extracted using encodec and phonemizer, respectively.
```
python phonemize_lrs.py \
    --save_dir "$SAVE_DIR_PATH" \
    --root_dir "$ROOT_DIR_PATH" \
    --encodec_model_path "$ENCODEC" \
    --split_name "$SPLIT_NAME"
```


### Detect the landmark from the video
Given the video, the facial landmarks are extracted.
```
python detect_landmark.py --root ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark \
 --manifest ${ROOT_DIR_PATH}/file.list --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}
```

### Extract mouth region from the video
Given the video and the extracted landmarks, extract the mouth region and save them.
```
python align_mouth.py --video-direc ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark --filename-path ${ROOT_DIR_PATH}/file.list \
 --save-direc ${SAVE_DIR_PATH}/video  --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}
```

### Construct the dataset and metadata for training
Given all the files, make the training dataset by concatenating source and target audio and text, and build a metadata for training.
```
#Make a continuation form for constructing training dataset
python construct_dataset.py --root_dir ${SAVE_DIR_PATH}
```







