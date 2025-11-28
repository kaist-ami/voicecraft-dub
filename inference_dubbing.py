import sys
sys.path.insert(0, 'fairseq')
from models import voicecraft_dub
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)
from data.lip_preprocess import detect_face_landmarks, align_mouth
import cv2
import argparse
import random
import numpy as np
import torchaudio
import torch
import os
import pickle
import logging
import glob
from tqdm import tqdm
import random
import json
import pdb
import subprocess
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def process_video(video_path):
    transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.421, 0.165)])
    frames = load_video(video_path)
    frames = transform(frames)
    frames = torch.FloatTensor(frames)
    return frames

def get_model(exp_dir, device=None):
    with open(os.path.join(exp_dir, "args.pkl"), "rb") as f:
        model_args = pickle.load(f)

    logging.info("load model weights...")
    model = voicecraft_dub.VoiceCraft_Dub(model_args)
    ckpt_fn = os.path.join(exp_dir, "best_bundle.pth")
    ckpt = torch.load(ckpt_fn, map_location='cpu')['model']
    phn2num = torch.load(ckpt_fn, map_location='cpu')['phn2num']
    model.load_state_dict(ckpt)
    del ckpt
    logging.info("done loading weights...")
    if device == None:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    return model, model_args, phn2num


def load_video(path):
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="VoiceCraft-Dub Inference")
    parser.add_argument("-st", "--silence_tokens", type=int, nargs="*",
                        default=[1388, 1898, 131], help="Silence token IDs")
    parser.add_argument("-casr", "--codec_audio_sr", type=int,
                        default=16000, help="Codec audio sample rate.")
    parser.add_argument("-csr", "--codec_sr", type=int, default=50,
                        help="Codec sample rate.")
    parser.add_argument("-n_codebooks", "--n_codebooks", type=int, default=4,
                        help="Codec sample rate.")
    parser.add_argument("-k", "--top_k", type=float,
                        default=0, help="Top k value.")
    parser.add_argument("-p", "--top_p", type=float,
                        default=0.8, help="Top p value.")
    parser.add_argument("-t", "--temperature", type=float,
                        default=1, help="Temperature value.")
    parser.add_argument("-kv", "--kvcache", type=float, choices=[0, 1],
                        default=0, help="Kvcache value.")
    parser.add_argument("-sr", "--stop_repetition", type=int,
                        default=-1, help="Stop repetition for generation")
    parser.add_argument("--sample_batch_size", type=int,
                        default=3, help="Batch size for sampling")
    parser.add_argument("-s", "--seed", type=int,
                        default=1, help="Seed value.")
    parser.add_argument("-bs", "--beam_size", type=int, default=50,
                        help="beam size for MFA alignment")

    parser.add_argument("-model_dir", "--model_dir", type=str, default="./pretrained_models/voicecraft_dub")
    parser.add_argument("-encodec_dir", "--encodec_dir", type=str, default="./pretrained_models/encodec.th")
    parser.add_argument("-face_preprocess_dir", "--face_preprocess_dir", type=str,default="./pretrained_models/landmarks")
    parser.add_argument("-result_dir", "--result_dir", type=str, default="./results")
    parser.add_argument("-src_text", "--src_text", type=str)
    parser.add_argument("-src_audio", "--src_audio", type=str)
    parser.add_argument("-tar_text", "--tar_text", type=str)
    parser.add_argument("-tar_vid", "--tar_vid", type=str)

    args = parser.parse_args()
    return args



def replace_audio_in_video(video_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,  # Input video file
        '-i', audio_path,  # Input audio file
        '-c:v', 'copy',  # Copy the video stream without re-encoding
        '-c:a', 'aac',  # Re-encode audio to AAC
        '-strict', 'experimental',
        '-map', '0:v',  # Map video stream from the video file
        '-map', '1:a',  # Map audio stream from the audio file
        output_path  # Output file
    ]
    subprocess.run(command, check=True)

def audio_decode(audio_tokenizer, root_dir, target_video):
    gen_list = glob.glob(os.path.join(root_dir, "gen*.npy"))
    for gen_path in gen_list:
        gen = np.load(gen_path)
        gen = torch.from_numpy(gen).cuda()
        if len(gen.shape) == 2:
            gen = gen.unsqueeze(0)
        gen_audio = audio_tokenizer.decode([(gen, None)])
        gen_audio = gen_audio[0].cpu()

        seg_save_fn_gen = os.path.join(root_dir, gen_path.split("/")[-1].replace(".npy", ".wav"))
        torchaudio.save(seg_save_fn_gen, gen_audio.detach(), 16000)
        video_file = target_video
        output_file = os.path.join(root_dir, gen_path.split("/")[-1].replace(".npy", ".mp4"))
        replace_audio_in_video(video_file, seg_save_fn_gen, output_file)



@torch.no_grad()
def inference_dubbing(model, model_args, original_audio, text_tokens, target_video,device, decode_config):
    text_tokens = torch.LongTensor(text_tokens)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    if decode_config['sample_batch_size'] >= 1:
        logging.info(f"running inference with batch size 1")
        concat_frames, gen_frames = model.inference_dubbing_batch(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[..., :model_args.n_codebooks].to(device),  # [1,T,8]
            target_video,
            top_k=decode_config['top_k'],
            top_p=decode_config['top_p'],
            temperature=decode_config['temperature'],
            stop_repetition=decode_config['stop_repetition'],
            kvcache=decode_config['kvcache'],
            silence_tokens=eval(decode_config['silence_tokens']) if type(
                decode_config['silence_tokens']) == str else
            decode_config['silence_tokens'],
            batch_size = decode_config['sample_batch_size']
        )  # output is [1,K,T]

    return concat_frames, gen_frames
def dubbing(args, audio_tokenizer, text_tokenizer, src_text_fn, tar_text_fn):
    # setup the hyperparameters for voicecraft-dub decoding
    codec_audio_sr = args.codec_audio_sr
    codec_sr = args.codec_sr
    top_k = args.top_k
    top_p = args.top_p  # defaults to 0.9 can also try 0.8, but 0.9 seems to work better
    temperature = args.temperature
    silence_tokens = args.silence_tokens
    kvcache = args.kvcache  # NOTE if OOM, change this to 0, or try the 330M model
    stop_repetition = args.stop_repetition
    sample_batch_size = args.sample_batch_size

    #define the model
    model, model_args, phn2num = get_model(args.model_dir)
    config = vars(model.args)
    model.to(device)

    #setup the paths for the source audio and target video
    src_audio_fn = args.src_audio
    tar_video_fn = args.tar_vid

    # encode audio
    encoded_frames = tokenize_audio(audio_tokenizer, src_audio_fn, offset=0)
    audio_tokens = encoded_frames[0][0].transpose(2, 1)  # [1,T,K]
    assert audio_tokens.ndim == 3 and audio_tokens.shape[0] == 1 and audio_tokens.shape[2] == model_args.n_codebooks, audio_tokens.shape

    ##extract and merge text token
    src_text = open(src_text_fn, "r").readline().split("Text: ")[1].strip().lower()
    tar_text = open(tar_text_fn, "r").readline().split("Text: ")[1].strip().lower()
    combine_text = src_text + ". " + tar_text

    text_tokens = [phn2num[phn] for phn in tokenize_text(text_tokenizer, text=combine_text.strip()) if phn in phn2num]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)

    ##extract_video
    detect_face_landmarks(tar_video_fn, args.face_preprocess_dir)
    input_video = align_mouth(tar_video_fn, args.face_preprocess_dir)
    target_video = process_video(input_video)

    # inference
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition,'kvcache': kvcache,"codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens,"sample_batch_size": sample_batch_size}
    concated_audio, gen_audio = inference_dubbing(model, argparse.Namespace(**config), audio_tokens,text_tokens, target_video, device=device,decode_config=decode_config)

    # save the audio and overlayed video
    os.makedirs(args.result_dir, exist_ok=True)
    save_dir = os.path.join(f"{args.result_dir}", input_video.split("/")[-2])
    os.makedirs(save_dir,exist_ok=True)
    for g_idx, gg in enumerate(gen_audio):
        np.save(os.path.join(save_dir, f"gen_{g_idx}.npy"), gg.cpu().numpy())
    audio_decode(audio_tokenizer, save_dir, tar_video_fn)


def main():
    args = parse_arguments()
    #define seed
    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)

    # setup text and audio tokenizer
    text_tokenizer = TextTokenizer(backend="espeak")
    encodec_fn = args.encodec_dir
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    # setup the source and target text paths
    src_text_fn = args.src_text
    tar_text_fn = args.tar_text

    dubbing(args, audio_tokenizer, text_tokenizer, src_text_fn, tar_text_fn)

if __name__ == '__main__':
    main()


