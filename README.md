# VoiceCraft-Dub: Automated Video Dubbing with Neural Codec Language Models (ICCV 2025)
[![Paper](https://img.shields.io/badge/arXiv-2504.02386-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.02386) [![Demo page](https://img.shields.io/badge/Project_Page-blue?logo=Github&style=flat-square)](https://voicecraft-dub.github.io/)


### TL;DR
VoiceCraft-Dub **synthesizes high-quality speech** from **text and facial cues**. Unlike text-to-speech, which generates diverse speech based on target text, VoiceCraft-Dub requires synthesized speech to be temporally and expressively aligned with the video while maintaining naturalness and intelligibility.
<img width="2316" height="701" alt="Image" src="https://github.com/user-attachments/assets/c216703b-a28a-4cc9-83e7-980facfa38b9" />

## Getting started
This code was developed on Ubuntu 18.04 with Python 3.9.16, CUDA 11.7 and PyTorch 2.0.1. 
Later versions should work, but have not been tested.

### Installation
Create and activate a virtual environment to work in:
```
conda create -n voicecraft_dub python=3.9.16
conda activate voicecraft_dub
```

Install the requirements with pip and [PyTorch](https://pytorch.org/). For CUDA 11.7, this would look like:
```
#Install xformers, audiocraft, and torch
pip install xformers==0.0.22
pip install git+https://github.com/dillionverma/audiocraft
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

#Installatoin for fairseq and AVHubert
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout afc77bd
pip install -e ./
cd ..

#Install rest of the requirements
pip install -r requirements.txt
```

### Download models
To run VoiceCraft-Dub, you need to download the pretrained models.

Download the following models:
VoiceCraft-Dub ([voicecraft_dub](https://drive.google.com/file/d/1HKsGQ4KyCpm2MJfaO3leYDi_9Ea7oKlL/view?usp=sharing)) | AVHubert ([large_lrs3_iter5.pt](https://facebookresearch.github.io/av_hubert/)) | Encodec ([encodec.th](https://drive.google.com/file/d/1_l7UGSrVZuwjy5JEQbTGtJdp0jTMytGL/view?usp=sharing)) | VoiceCraft ([giga830.pth](https://drive.google.com/file/d/1-yGRxsC_C9DFJPYPHtClRurDtczxlq-U/view?usp=sharing)) | [landmarks](https://drive.google.com/file/d/1ALDnUXrmJ0W7J1UjmmupXsugG3s89tem/view?usp=sharing)

**VoiceCraft-Dub** model is our trained model, while **AVHubert** is used for extracting lip features from the facial video. **Encodec** is required for encoding and decoding the audio, weights in **landmark** directory are used for lip preprocessing, and **VoiceCraft** is used for initializing the model when training.


After downloading the models, place them in `./pretrained_models`.
```
./pretrained_models/avhubert/large_lrs3_iter5.pt
./pretrained_models/voicecraft_dub
./pretrained_models/giga830.pth
./pretrained_models/encodec.th
```

### Download dataset
For checking the code, we provide sample source data and the preprocessed data in ./samples.
```
./samples/trainval #source data from LRS3
./samples/trainval_preprocess #preprocessed data for training
./samples/test #samples for testing
```

For training the model from scratch, downloading the whole dataset is required.
We used two different datasets for training and validating VoiceCraft-Dub.
Download [LRS3](https://mmai.io/datasets/lip_reading/) and [CelebV-Dub](https://drive.google.com/file/d/1pL4C4sRiQimbsFHXQKYx6NNyjOhPvR3l/view?usp=sharing).


## How to run inference
Run below command to generate speech from both text and facial video.
For the inference, total four files are required: **source transcript**, **target transcript**, **source audio**, and **target facial video**.

We provide samples for the inference in ./samples directory.
```
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

#or simply run

sh inference.sh
```

## How to preprocess the training dataset
To preprocess the dataset, the whole training dataset from [LRS3](https://mmai.io/datasets/lip_reading/) or [CelebV-Dub](https://drive.google.com/file/d/1pL4C4sRiQimbsFHXQKYx6NNyjOhPvR3l/view?usp=sharing) should be downloaded first.
We provide the sample data in ./samples directory.

Given the **./samples/trainval**, which is the source dataset, the following code preprocesses the data and save them as **./samples/trainval_preprocess**.
```
cd data
sh preprocess_data.sh
```
More details are in [README.md](https://github.com/kaist-ami/voicecraft-dub/tree/main/data) in the ./data directory. 


## How to run training
To train the VoiceCraft-Dub model, you need to first follow the aforementioned process for downloading the pretrained models and datasets, and preprocessing the training dataset.
Then, run the following code.
```
sh z_scripts/train.sh
```

## License
The codebase is under CC BY-NC-SA 4.0 ([LICENSE-CODE](./LICENSE-CODE)), and the model weights are under Coqui Public Model License 1.0.0 ([LICENSE-MODEL](./LICENSE-MODEL)). Note that we use some of the code from other repository that are under different licenses: `./models/codebooks_patterns.py` is under MIT license; `./models/modules`, `./steps/optim.py`, `data/tokenizer.py` are under Apache License, Version 2.0; the phonemizer we used is under GNU 3.0 License.


## **Acknowledgement**
We heavily borrow the code from [VoiceCraft](https://github.com/jasonppy/VoiceCraft) and [VALL-E reproduction](https://github.com/lifeiteng/vall-e), and dataset from [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ) and [CelebV-Text](https://celebv-text.github.io/). We sincerely appreciate those authors.

## Citation
```
@article{sung2025voicecraft,
  title={VoiceCraft-Dub: Automated Video Dubbing with Neural Codec Language Models},
  author={Sung-Bin, Kim and Choi, Jeongsoo and Peng, Puyuan and Chung, Joon Son and Oh, Tae-Hyun and Harwath, David},
  journal={arXiv preprint arXiv:2504.02386},
  year={2025}
}
```

## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

