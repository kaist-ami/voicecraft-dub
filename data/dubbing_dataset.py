# cp from https://github.com/jasonppy/VoiceCraft modified by Sungbin Kim

import os
import torch
import random
import copy
import logging
import shutil
import pdb
import sys
import cv2
import numpy as np

class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'validation', 'test']
        manifest_fn = os.path.join(self.args.dataset_dir, self.args.manifest_name, self.split+".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[-1]) for item in data]

        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,self.args.vocab_name)
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, self.args.vocab_name))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}

        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])
        self.read_split_length()

        self.video_path = os.path.join(self.args.dataset_dir, "video")
        self.image_crop_size = 88

        self.image_mean, self.image_std = 0.421, 0.165
        self.transform = Compose([
            Normalize(0.0, 255.0),
            CenterCrop((self.image_crop_size, self.image_crop_size)),
            Normalize(self.image_mean, self.image_std)])


    def read_split_length(self):
        self.split_dic={}
        split_txt = open(os.path.join(self.args.dataset_dir, "split_len.txt"),"r")
        for line in split_txt:
            txt_name = line.split(",")[0]
            length = int(line.split(",")[1].strip())
            self.split_dic[txt_name]=length

    def process_video(self, video_path):
        frames = self.load_video(video_path)
        frames = self.transform(frames)
        frames = torch.FloatTensor(frames)
        return frames

    def load_video(self, path):
        for i in range(3):
            try:
                cap = cv2.VideoCapture(path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.append(frame)
                    else:
                        break
                frames = np.stack(frames)
                return frames
            except Exception:
                print(f"failed loading {path} ({i} / 3)")
                if i == 2:
                    raise ValueError(f"Unable to load {path}")

    def __len__(self):
        return len(self.lengths_list)

    def pad_video(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size] + audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False)  #
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff] + audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2)  # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute(
                (0, 4, 1, 2, 3)).contiguous()  # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def _load_phn_enc(self, index):
        item = self.data[index]
        pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
        ef = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item[1]+".txt")
        # try:
        with open(pf, "r") as p, open(ef, "r") as e:
            phns = [l.strip() for l in p.readlines()]
            assert len(phns) == 1, phns
            x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
            encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]

            assert len(encos) == self.args.n_codebooks, ef
            if self.args.special_first:
                y = [[int(n)+self.args.n_special for n in l] for l in encos]
            else:
                y = [[int(n) for n in l] for l in encos]

        split_length = self.split_dic[item[1]+".txt"]

        return x, y, split_length

    def __getitem__(self, index):
        while 1:
            try:
                x, y, split_length = self._load_phn_enc(index)
                break
            except:
                index = random.choice(range(len(self)))  # regenerate an index
                continue

        x_len, y_len = len(x), len(y[0])

        if x_len == 0 or y_len == 0:
            return {
            "x": None,
            "x_len": None,
            "y": None,
            "y_len": None,
            "y_mask_interval": None, # index y_mask_interval[1] is the position of start_of_continue token
            "extra_mask_start": None # this is only used in VE1
            }

        while y_len < self.args.encodec_sr*self.args.audio_min_length:
            try:
                index = random.choice(range(len(self))) # regenerate an index
                x, y, split_length = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])
            except:
                continue

        if self.args.drop_long:
            while x_len > self.args.text_max_length or y_len > self.args.encodec_sr*self.args.audio_max_length:
                try:
                    index = random.choice(range(len(self))) # regenerate an index
                    x, y,split_length = self._load_phn_enc(index)
                    x_len, y_len = len(x), len(y[0])
                except:
                    continue

        ### padding and cropping below ###
        ### padding and cropping below ###
        # adjust the length of encodec codes, pad to max_len or randomly crop
        orig_y_len = copy.copy(y_len)
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)

        if y_len > max_len:

            audio_start = random.choice(range(0, y_len-max_len))
            for i in range(len(y)):
                y[i] = y[i][audio_start:(audio_start+max_len)]
            y_len = max_len
        else:
            audio_start = 0
            if not self.args.dynamic_batching:
                pad = [0] * (max_len - y_len) if self.args.sep_special_token else [self.args.audio_pad_token] * (max_len - y_len)
                for i in range(len(y)):
                    y[i] = y[i] + pad

        # adjust text
        # if audio is cropped, and text is longer than max, crop max based on how audio is cropped
        if audio_start > 0 and len(x) > self.args.text_max_length: # if audio is longer than max and text is long than max, start text the way audio started

            x = x[int(len(x)*audio_start/orig_y_len):]
            if len(x) > self.args.text_max_length: # if text is still longer than max, cut the end
                x = x[:self.args.text_max_length]

        x_len = len(x)
        if x_len > self.args.text_max_length:

            text_start = random.choice(range(0, x_len - self.args.text_max_length))
            x = x[text_start:text_start+self.args.text_max_length]
            x_len = self.args.text_max_length
        elif self.args.pad_x and x_len <= self.args.text_max_length:
            pad = [0] * (self.args.text_max_length - x_len) if self.args.sep_special_token else [self.args.text_pad_token] * (self.args.text_max_length - x_len)
            x = x + pad

        data_id = self.data[index][1].split("__")[0]

        if self.args.dataset_name=="CELEB":
            try:
                vid = self.process_video(
                    os.path.join(self.video_path, data_id + "_" + self.data[index][1].split("__")[-1] + ".mp4"))
            except:
                data_id = "__".join(self.data[index][1].split("__")[:2])
                vid = self.process_video(
                    os.path.join(self.video_path, data_id + "_" + self.data[index][1].split("__")[-1] + ".mp4"))
        else:
            vid = self.process_video(os.path.join(self.video_path, self.data[index][1].split("__")[0]+"_"+self.data[index][1].split("__")[-1] + ".mp4"))
        vid_lens = vid.shape[0]

        return {
            "x": torch.LongTensor(x),  # phoneme
            "x_len": x_len,
            "y": torch.LongTensor(y),  # encodec
            "y_len": y_len,
            "split_length": split_length,
            "vid": vid,
            "vid_lens":vid_lens
        }


    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else: # goes here
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2: #goes here
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)

        res['vid'] = torch.nn.utils.rnn.pad_sequence([item for item in out['vid']],batch_first=True)
        res['vid_lens'] = torch.LongTensor(out["vid_lens"])
        res["vid_padding_mask"] = torch.arange(res['vid'][0].shape[0]).unsqueeze(0) >= res['vid_lens'].unsqueeze(1)

        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)

        res["split_length"] = torch.ShortTensor(out["split_length"])

        return res