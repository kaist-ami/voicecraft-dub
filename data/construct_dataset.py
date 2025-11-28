import os
import glob
import pdb
import shutil
from tqdm import tqdm
import argparse
def count_numbers_in_file(filename):
    total_count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Split the line into a list of numbers as strings
                numbers = line.split()
                # Increment the total count by the number of elements in the list
                total_count += len(numbers)
                break
    except Exception as e:
        # Split the line into a list of numbers as strings
        numbers = filename.split()
        # Increment the total count by the number of elements in the list
        total_count += len(numbers)
    except FileNotFoundError:
        print("The file was not found.")
    return total_count


def save_train_txt(files,root_dir):
    save_txt = open(os.path.join(root_dir, "manifest", "train.txt"), "w")

    for ff in tqdm(files):
        file_name = ff.split("/")[-1].split(".")[0]
        total_cnt = count_numbers_in_file(ff)
        save_txt.write("0\t"+file_name+"\t"+str(total_cnt)+"\n")
    save_txt.close()

def save_files(file_lists_temp, target_dir, speaker_utt):
    split_len_txt = open(os.path.join(target_dir, f"split_len.txt"), "a")
    target_text_dir = os.path.join(target_dir, "phonemes")
    target_encodec_dir = os.path.join(target_dir, "encodec_16khz_4codebooks")

    file_lists_temp = file_lists_temp
    for ff in tqdm(file_lists_temp):
        speaker = "_".join(ff.split("_")[:-1])
        utt_id = ff.split("_")[-1].split(".")[0]

        for utt in speaker_utt[speaker]:
            if utt==utt_id:continue
            new_ff = speaker+"_"+utt+".txt"
            save_name = speaker+"__"+utt_id+"__"+utt+".txt"

            #merge encodec
            orig_emb=open(os.path.join(target_dir, "encodec_16khz_4codebooks_", ff), "r")
            target_emb = open(os.path.join(target_dir, "encodec_16khz_4codebooks_", new_ff), "r")
            new_encodec = open(os.path.join(target_encodec_dir, save_name), "w")

            for orig, target in zip(orig_emb, target_emb):
                len_orig = count_numbers_in_file(orig.strip())
                new_code = orig.strip()+" "+target.strip()
                new_encodec.write(new_code+"\n")
            new_encodec.close()
            split_len_txt.write(save_name+","+str(len_orig)+"\n")

            # merge text
            orig_phone = open(os.path.join(target_dir, "phonemes_", ff), "r")
            target_phone = open(os.path.join(target_dir, "phonemes_", new_ff), "r")
            new_phonemes = open(os.path.join(target_text_dir, save_name), "w")

            for orig, target in zip(orig_phone, target_phone):
                new_phoneme = orig[:-1]+target
                new_phonemes.write(new_phoneme)

            new_phonemes.close()
    split_len_txt.close()


parser = argparse.ArgumentParser(description='construct dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root_dir', type=str, help='root dir')
args = parser.parse_args()

video_source_dir = os.path.join(args.root_dir, "video")
source_dir = args.root_dir
target_dir = args.root_dir
target_encodec_dir = os.path.join(target_dir, "encodec_16khz_4codebooks")
target_text_dir = os.path.join(target_dir, "phonemes")

os.makedirs(target_dir, exist_ok=True)
os.makedirs(target_encodec_dir, exist_ok=True)
os.makedirs(target_text_dir, exist_ok=True)

file_lists = os.listdir(os.path.join(source_dir, "encodec_16khz_4codebooks_"))
speaker_utt={}
for ff in file_lists:
    speaker="_".join(ff.split("_")[:-1])
    utt_id = ff.split("_")[-1].split(".")[0]

    if speaker not in speaker_utt.keys():
        speaker_utt[speaker]=[utt_id]
    else:
        speaker_utt[speaker].append(utt_id)

file_lists=sorted(file_lists)
save_files(file_lists, target_dir, speaker_utt)


root_dir = args.root_dir
source_dir = os.path.join(root_dir, "encodec_16khz_4codebooks")
file_list = glob.glob(os.path.join(source_dir,"*.txt"))
os.makedirs(os.path.join(root_dir, "manifest"), exist_ok=True)
save_train_txt(file_list, root_dir)




