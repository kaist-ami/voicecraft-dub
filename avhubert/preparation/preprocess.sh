vox="/saltpool0/data/sungbin/CVPR24/lip2speech/1017_sanity"
#1. Preparing dataset
#python celeb_prepare.py --root ${vox} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
#python celeb_prepare.py  --root ${vox} --ffmpeg ffmpeg --rank 1 --step 2

##2. Detect facial landmark and crop mouth ROIs:
python detect_landmark.py --root ${vox} --landmark ${vox}/landmark --manifest ${vox}/file.list \
 --ffmpeg ffmpeg \
 --rank 1 --nshard 1

#python align_mouth.py --video-direc ${vox}/original --landmark ${vox}/landmark --filename-path ${vox}/file.list \
# --save-direc ${vox}/video  --ffmpeg ffmpeg \
# --rank 1 --nshard 1


##3. Count number of frames
#python count_frames.py --root ${vox} --manifest ${vox}/file.list --nshard 1 --rank 1

#
##4. Set dataset directory
#python celeb_manifest.py --vox ${vox} \
# --en-ids /path/to/en