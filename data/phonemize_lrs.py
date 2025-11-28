import argparse
import pdb
import glob
import librosa
import re

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument('--save_dir', type=str, default="../samples/trainval_preprocess")
    parser.add_argument('--root_dir', type=str, default="../samples/trainval")
    parser.add_argument('--split_name', type=str, default="../samples/trainval")
    parser.add_argument('--encodec_model_path', type=str,default="../pretrained_models/encodec.th")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=100,
                        help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    parser.add_argument('--max_len', type=int, default=30000,
                        help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
    return parser.parse_args()

if __name__ == "__main__":
    import logging

    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time
    from datasets import load_dataset, DownloadConfig

    from tokenizer import TextTokenizer, tokenize_text

    # get the path
    phn_save_root = os.path.join(args.save_dir, "phonemes_")
    codes_save_root = os.path.join(args.save_dir, "encodec_16khz_4codebooks_")
    vocab_fn = os.path.join(args.save_dir, "vocab.txt")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)

    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]] * args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]] * args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(
            f"median: {lens[inds[len(inds) // 2]] * args.model_code_sr} encodec codes, {lens[inds[len(inds) // 2]]:.2f} sec.")
        logging.info(
            f"95 percentile longest: {lens[inds[int(len(inds) * 0.95)]] * args.model_code_sr} encodec codes, {lens[inds[int(len(inds) * 0.95)]]:.2f} sec.")
        return inds[::-1]


    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a)) + '\n')
            f.write(' '.join(map(str, array[-1])))

    ### phonemization
    from audiocraft.solvers import CompressionSolver

    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()
    text_tokenizer = TextTokenizer()

    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?",
                " <EXCLAMATIONPOINT>": "!"}  # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#",
               "<OTHER>": "%#%"}  # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = {"h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>",
                "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    stime = time.time()
    logging.info("loading the dataset...")

    split = args.split_name
    txt_file_list = glob.glob(os.path.join(args.root_dir, split, "*", "*.txt"))
    phn_vocab = set()
    all_lens = []

    ######################save the text into phonemes #################################
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not a issue
    skip = 0

    logging.info(f"now processing split {split}...")

    for item in tqdm.tqdm(txt_file_list):

        id_name = item.split("/")[-2]
        utt_name = item.split("/")[-1].split(".")[0]
        save_name = id_name + "_" + utt_name
        save_fn = os.path.join(phn_save_root, save_name + ".txt")

        # txt_file = open(item.replace(".wav", ".txt"), "r")
        txt_file = open(item, "r")

        text = re.sub(r'[^a-zA-Z]+$', '', txt_file.readline().split("Text:")[1].strip()) + " <PERIOD>"
        if sum(word in forbidden_words for word in text.split(" ")):
            logging.info(f"skip {item['segment_id']}, because it contains forbiden words. It's transcript: {text}")
            skip += 1
            continue
        for k, v in punc2sym.items():
            text = text.replace(k, v)
        phn = tokenize_text(text_tokenizer, text)
        phn_seq = " ".join(phn)
        for k, v in word2sym.items():
            phn_seq = phn_seq.replace(k, v)
        phn_vocab.update(phn_seq.split(" "))
        all_lens.append(len(phn_seq.split(" ")))
        with open(save_fn, "w") as f:
            f.write(phn_seq)

    logging.info(f"split {split} has {len(txt_file_list)} samples in total, skipped {skip} due to forbiden words")

    print(f"phn vocab size: {len(list(phn_vocab))}")
    print("phn sequence stats: ")
    print(f"longest: {max(all_lens)}")
    print(f"shortest: {min(all_lens)}")
    print(f"median: {np.quantile(all_lens, 0.5)}")
    print(f"95 percentile longest: {np.quantile(all_lens, 0.95)}")
    print("write vocabulary to ", vocab_fn)

    with open(vocab_fn, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")

    class mydataset(torch.utils.data.Dataset):
        def __init__(self):
            super().__init__()
            self.data = txt_file_list
        def __len__(self):
            return len(self.data)

        def load_audio(self, audio_path):
            audio_file, sr = librosa.load(audio_path.replace(".txt",".wav"), sr=16000)
            duration = librosa.get_duration(y=audio_file, sr=16000)
            return torch.from_numpy(audio_file).float(), duration

        def __getitem__(self, ind):
            # try:
            id_name = self.data[ind].split("/")[-2]
            utt_name = self.data[ind].split("/")[-1].split(".")[0]
            segment_id = id_name+"_"+utt_name
            audio, duration = self.load_audio(self.data[ind])
            sr = 16000

            txt_file = open(self.data[ind].replace(".wav", ".txt"), "r")
            text = txt_file.readline().split("Text:")[1].strip() + " <PERIOD>"
            # except:
            #     return None, None, None, None, None, None

            return segment_id, audio, sr, text, duration

        def collate(self, batch):
            res = {'segment_id': [], "audio": [], "sr": [], "text": [], "duration": []}
            for item in batch:
                if item[0] != None:
                    res['segment_id'].append(item[0])
                    res['audio'].append(item[1])
                    res['sr'].append(item[2])
                    res['text'].append(item[3])
                    res['duration'].append(item[4])
            return res

    ## encodec codes extraction
    logging.info("encodec encoding...")

    test_dataset = mydataset()
    test_loader = torch.torch.utils.data.DataLoader(test_dataset, batch_size=args.mega_batch_size, shuffle=False,
                                                    drop_last=False, num_workers=args.n_workers,
                                                    collate_fn=test_dataset.collate)

    splits = [args.split_name]
    loaders = [test_loader]
    for split, loader in zip(splits, loaders):
        skip = 0
        logging.info(f"now processing split {split}...")

        mega_n_steps = int(np.ceil(len(txt_file_list) / args.mega_batch_size))
        logging.info(f"partition the split {split} into {mega_n_steps} parts, each has {args.mega_batch_size} samples")
        for m, mega_batch in enumerate(loader):

            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m + 1}/{mega_n_steps}")
            lengths = np.array(mega_batch['duration'])

            sorted_inds = sort_by_audio_len(lengths)
            for j in range(len(sorted_inds))[::-1]:
                if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap:  # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]

            n_steps = int(np.ceil(len(sorted_inds) / args.batch_size))
            for n in tqdm.tqdm(range(n_steps), disable=True):
                inds_used = sorted_inds[n * args.batch_size:(n + 1) * args.batch_size]
                audio_batch = [mega_batch['audio'][id] for id in inds_used]
                sr_batch = [mega_batch['sr'][id] for id in inds_used]
                segment_id_batch = [mega_batch['segment_id'][id] for id in inds_used]
                text_batch = [mega_batch['text'][id] for id in inds_used]
                padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(
                    1)  # [B, T] -> [B, 1, T]
                all_lens = [lengths[id] for id in inds_used]
                with torch.no_grad():
                    if max(all_lens) > args.max_len and len(
                            all_lens) > 1:  # NOTE decrease args.max_len if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model.encode(inwav[:len(inwav) // 2])[0].cpu())
                        codes.append(model.encode(inwav[len(inwav) // 2:])[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model.encode(padded_wav.cuda())
                        # logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()

                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(codes_save_root, segment_id_batch[i] + ".txt")
                    if os.path.isfile(save_fn): continue
                    actual_len = round(length * args.model_code_sr)  # 320 is downsample rate for this model
                    cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                    write_array_to_txt_file(cur_code, save_fn)
