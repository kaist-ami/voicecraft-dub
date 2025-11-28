# from typing import List, Dict
# from pathlib import Path
# import argparse
#
# import numpy as np
# import torch
# from torch import nn
# from skimage import io
# from face_alignment.detection.sfd.sfd_detector import SFDDetector
#
# from emonet.models import EmoNet
#
# import cv2
# import pdb
# import os
# import glob
# from tqdm import tqdm
# import shutil
# def load_video(video_path: Path) -> List[np.ndarray]:
#     """
#     Loads a video using OpenCV.
#     """
#     video_capture = cv2.VideoCapture(video_path)
#
#     list_frames_rgb = []
#
#     # Reads all the frames
#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#
#         if not ret:
#             break
#
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         list_frames_rgb.append(image_rgb)
#
#     return list_frames_rgb
#
#
# def load_emonet(n_expression: int, device: str):
#     """
#     Loads the emotion recognition model.
#     """
#
#     # Loading the model
#     # state_dict_path = Path(__file__).parent.joinpath(
#     #     "pretrained", f"emonet_{n_expression}.pth"
#     # )
#     #state_dict_path = "/home/sungbin/CVPR24/emotion_rec/pretrained/emonet_8.pth"
#     state_dict_path = "./emonet_8.pth"
#     print(f"Loading the emonet model from {state_dict_path}.")
#     state_dict = torch.load(str(state_dict_path), map_location="cpu")
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     net = EmoNet(n_expression=n_expression).to(device)
#     net.load_state_dict(state_dict, strict=False)
#     net.eval()
#
#     return net
#
#
# def run_emonet(
#     emonet: torch.nn.Module, frame_rgb: np.ndarray
# ) -> Dict[str, torch.Tensor]:
#     """
#     Runs the emotion recognition model on a single frame.
#     """
#     # Resize image to (256,256)
#     image_rgb = cv2.resize(frame_rgb, (image_size, image_size))
#
#     # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
#     image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0
#     with torch.no_grad():
#         output = emonet(image_tensor.unsqueeze(0))
#
#     return output
#
#
# def plot_valence_arousal(
#     valence: float, arousal: float, circumplex_size=512
# ) -> np.ndarray:
#     """
#     Assumes valence and arousal in range [-1;1].
#     """
#     circumplex_path = Path(__file__).parent / "images/circumplex.png"
#
#     circumplex_image = cv2.imread(circumplex_path)
#     circumplex_image = cv2.resize(circumplex_image, (circumplex_size, circumplex_size))
#
#     # Position in range [0,circumplex_size/2] - arousal axis goes up, so need to take the opposite
#     position = (
#         (valence + 1.0) / 2.0 * circumplex_size,
#         (1.0 - arousal) / 2.0 * circumplex_size,
#     )
#
#     cv2.circle(
#         circumplex_image, (int(position[0]), int(position[1])), 16, (0, 0, 255), -1
#     )
#
#     return circumplex_image
#
#
# def make_visualization(
#     frame_rgb: np.ndarray,
#     face_crop_rgb: np.ndarray,
#     face_bbox: torch.Tensor,
#     emotion_prediction: Dict[str, torch.Tensor],
#     font_scale=2,
# ) -> np.ndarray:
#     """
#     Composes the final visualization with detected face, landmarks, discrete and continuous emotions.
#     """
#     # Visualize the detected face
#     cv2.rectangle(
#         frame_rgb,
#         (face_bbox[0], face_bbox[1]),
#         (face_bbox[2], face_bbox[3]),
#         (255, 0, 0),
#         8,
#     )
#
#     # Add the discrete emotion next to it
#     predicted_emotion_class_idx = (
#         torch.argmax(nn.functional.softmax(emotion_prediction["expression"], dim=1))
#         .cpu()
#         .item()
#     )
#     frame_rgb = cv2.putText(
#         frame_rgb,
#         emotion_classes[predicted_emotion_class_idx],
#         ((face_bbox[0] + face_bbox[2]) // 2, face_bbox[1] + 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 0, 0),
#         2,
#         cv2.LINE_AA,
#     )
#
#     # Landmarks visualization
#     # Resize to the original face_crop image size
#     heatmap = torch.nn.functional.interpolate(
#         emotion_prediction["heatmap"],
#         (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
#         mode="bilinear",
#     )
#
#     landmark_visualization = face_crop_rgb.copy()
#     for landmark_idx in range(heatmap[0].shape[0]):
#         # Detect the position of each landmark and draw a circle there
#         landmark_position = (
#             heatmap[0, landmark_idx, :, :] == torch.max(heatmap[0, landmark_idx, :, :])
#         ).nonzero()
#         cv2.circle(
#             landmark_visualization,
#             (
#                 int(landmark_position[0][1].cpu().item()),
#                 int(landmark_position[0][0].cpu().item()),
#             ),
#             4,
#             (255, 255, 255),
#             -1,
#         )
#
#     # Valence and arousal visualization
#     circumplex_bgr = plot_valence_arousal(
#         emotion_prediction["valence"].clamp(-1.0, 1.0),
#         emotion_prediction["arousal"].clamp(-1.0, 1.0),
#         frame_rgb.shape[0],
#     )
#
#     # Compose the final visualization
#     visualization = np.zeros(
#         (frame_rgb.shape[0], frame_rgb.shape[1] + frame_rgb.shape[0] // 2, 3),
#         dtype=np.uint8,
#     )
#
#     # Resize the circumplex and face crop to match the frame size
#     circumplex_bgr = cv2.resize(
#         circumplex_bgr, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
#     )
#     landmark_visualization = cv2.resize(
#         landmark_visualization, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
#     )
#     visualization[:, : frame_rgb.shape[1], :] = frame_rgb[:, :, ::-1].astype(np.uint8)
#     visualization[
#         : frame_rgb.shape[0] // 2, frame_rgb.shape[1] :, :
#     ] = landmark_visualization[:, :, ::-1].astype(
#         np.uint8
#     )  # OpenCV needs BGR
#     visualization[frame_rgb.shape[0] // 2 :, frame_rgb.shape[1] :, :] = (
#         circumplex_bgr.astype(np.uint8)
#     )
#
#     return visualization
#
#
# if __name__ == "__main__":
#
#     torch.backends.cudnn.benchmark = True
#
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--nclasses",
#         type=int,
#         default=8,
#         choices=[5, 8],
#         help="Number of emotional classes to test the model on. Please use 5 or 8.",
#     )
#     parser.add_argument(
#         "--video_path",
#         type=str,
#         default="video.mp4",
#         help="Path to a video.",
#     )
#     parser.add_argument(
#         "--output_path",
#         type=str,
#         default="output.mp4",
#         help="Path where the output video is saved.",
#     )
#
#     args = parser.parse_args()
#
#     # Parameters of the experiments
#     n_expression = args.nclasses
#     device = "cuda:0"
#     image_size = 256
#     emotion_classes = {
#         0: "Neutral",
#         1: "Happy",
#         2: "Sad",
#         3: "Surprise",
#         4: "Fear",
#         5: "Disgust",
#         6: "Anger",
#         7: "Contempt",
#     }
#
#     print(f"Loading emonet")
#     emonet = load_emonet(n_expression, device)
#
#     print(f"Loading face detector")
#     sfd_detector = SFDDetector(device)
#
#     print(f"Loading video")
#     video_dir = "/saltpool0/data/sungbin/CVPR24/voicecraft/1107_dataset_tts/xs/face/"
#     save_dir = "/saltpool0/data/sungbin/CVPR24/voicecraft/1107_dataset_tts/xs/face_npy"
#     os.makedirs(save_dir, exist_ok=True)
#     video_lists = sorted(glob.glob(os.path.join(video_dir,"*.mp4")))[400000:]
#     # video_path = Path(__file__).parent / args.video_path
#     id_dict={}
#     for vv in tqdm(video_lists):
#         id=vv.split("/")[-1].split("_")[0]
#         utt = vv.split("/")[-1].split("_")[2]
#         video_save_path = os.path.join(save_dir, vv.split("/")[-1].replace(".mp4",".npy"))
#
#         if id not in id_dict.keys():
#             id_dict[id]=[utt]
#         else:
#             if utt in id_dict[id]:
#                 shutil.copy(glob.glob(os.path.join(save_dir, id+"*"+utt.replace(".mp4",".npy")))[0],video_save_path)
#                 continue
#             else:
#                 id_dict[id].append(utt)
#
#         list_frames_rgb = load_video(vv)
#         visualization_frames = []
#
#         for i, frame in enumerate(list_frames_rgb):
#
#             # Run face detector
#             with torch.no_grad():
#                 # Face detector requires BGR frame
#                 detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])
#
#             # If at least a face has been detected, run emotion recognition on the first face
#             if len(detected_faces)>0:
#                 # Only take the first detected face
#                 bbox = np.array(detected_faces[0]).astype(np.int32)
#                 bbox[bbox<0]=0
#
#                 face_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
#
#                 image_rgb = cv2.resize(face_crop, (image_size, image_size))
#
#                 image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#                 visualization_frames.append(image_rgb)
#
#                 temp_rgb=image_rgb
#                 # emotion_prediction = run_emonet(emonet, face_crop.copy())
#                 #
#                 # visualization_bgr = make_visualization(
#                 #     frame.copy(), face_crop.copy(), bbox, emotion_prediction
#                 # )
#                 # visualization_frames.append(visualization_bgr)
#             else:
#                 # Visualization without emotion
#                 # visualization = np.zeros(
#                 #     (frame.shape[0], frame.shape[1] + frame.shape[0] // 2, 3),
#                 #     dtype=np.uint8,
#                 # )
#                 # visualization[:, : frame.shape[1], :] = frame[:, :, ::-1].astype(np.uint8)
#                 #
#                 # visualization_frames.append(visualization)
#                 visualization_frames.append(temp_rgb)
#         visualization_frames = np.array(visualization_frames)
#         image_tensor = torch.Tensor(visualization_frames).permute(0,3, 1, 2).to(device) / 255.0
#         with torch.no_grad():
#             output = emonet(image_tensor)
#         np.save(video_save_path, output.detach().cpu().numpy())
#



from typing import List, Dict
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from skimage import io
from face_alignment.detection.sfd.sfd_detector import SFDDetector

from emonet.models import EmoNet
import multiprocessing as mp
import cv2
import pdb
import os
import glob
from tqdm import tqdm
import shutil
import json

def load_video(video_path: Path) -> List[np.ndarray]:
    """
    Loads a video using OpenCV.
    """
    video_capture = cv2.VideoCapture(video_path)

    list_frames_rgb = []

    # Reads all the frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        list_frames_rgb.append(image_rgb)

    return list_frames_rgb


def load_emonet(n_expression: int, device: str):
    """
    Loads the emotion recognition model.
    """

    # Loading the model
    # state_dict_path = Path(__file__).parent.joinpath(
    #     "pretrained", f"emonet_{n_expression}.pth"
    # )
    #state_dict_path = "/home/sungbin/CVPR24/emotion_rec/pretrained/emonet_8.pth"
    state_dict_path = "/scratch/10441/sk63833/pretrained_models/emonet/emonet_8.pth"
    print(f"Loading the emonet model from {state_dict_path}.")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    return net


def run_emonet(
    emonet: torch.nn.Module, frame_rgb: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Runs the emotion recognition model on a single frame.
    """
    # Resize image to (256,256)
    image_rgb = cv2.resize(frame_rgb, (image_size, image_size))

    # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0
    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))

    return output


def plot_valence_arousal(
    valence: float, arousal: float, circumplex_size=512
) -> np.ndarray:
    """
    Assumes valence and arousal in range [-1;1].
    """
    circumplex_path = Path(__file__).parent / "images/circumplex.png"

    circumplex_image = cv2.imread(circumplex_path)
    circumplex_image = cv2.resize(circumplex_image, (circumplex_size, circumplex_size))

    # Position in range [0,circumplex_size/2] - arousal axis goes up, so need to take the opposite
    position = (
        (valence + 1.0) / 2.0 * circumplex_size,
        (1.0 - arousal) / 2.0 * circumplex_size,
    )

    cv2.circle(
        circumplex_image, (int(position[0]), int(position[1])), 16, (0, 0, 255), -1
    )

    return circumplex_image


def make_visualization(
    frame_rgb: np.ndarray,
    face_crop_rgb: np.ndarray,
    face_bbox: torch.Tensor,
    emotion_prediction: Dict[str, torch.Tensor],
    font_scale=2,
) -> np.ndarray:
    """
    Composes the final visualization with detected face, landmarks, discrete and continuous emotions.
    """
    # Visualize the detected face
    cv2.rectangle(
        frame_rgb,
        (face_bbox[0], face_bbox[1]),
        (face_bbox[2], face_bbox[3]),
        (255, 0, 0),
        8,
    )

    # Add the discrete emotion next to it
    predicted_emotion_class_idx = (
        torch.argmax(nn.functional.softmax(emotion_prediction["expression"], dim=1))
        .cpu()
        .item()
    )
    frame_rgb = cv2.putText(
        frame_rgb,
        emotion_classes[predicted_emotion_class_idx],
        ((face_bbox[0] + face_bbox[2]) // 2, face_bbox[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Landmarks visualization
    # Resize to the original face_crop image size
    heatmap = torch.nn.functional.interpolate(
        emotion_prediction["heatmap"],
        (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
        mode="bilinear",
    )

    landmark_visualization = face_crop_rgb.copy()
    for landmark_idx in range(heatmap[0].shape[0]):
        # Detect the position of each landmark and draw a circle there
        landmark_position = (
            heatmap[0, landmark_idx, :, :] == torch.max(heatmap[0, landmark_idx, :, :])
        ).nonzero()
        cv2.circle(
            landmark_visualization,
            (
                int(landmark_position[0][1].cpu().item()),
                int(landmark_position[0][0].cpu().item()),
            ),
            4,
            (255, 255, 255),
            -1,
        )

    # Valence and arousal visualization
    circumplex_bgr = plot_valence_arousal(
        emotion_prediction["valence"].clamp(-1.0, 1.0),
        emotion_prediction["arousal"].clamp(-1.0, 1.0),
        frame_rgb.shape[0],
    )

    # Compose the final visualization
    visualization = np.zeros(
        (frame_rgb.shape[0], frame_rgb.shape[1] + frame_rgb.shape[0] // 2, 3),
        dtype=np.uint8,
    )

    # Resize the circumplex and face crop to match the frame size
    circumplex_bgr = cv2.resize(
        circumplex_bgr, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    landmark_visualization = cv2.resize(
        landmark_visualization, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    visualization[:, : frame_rgb.shape[1], :] = frame_rgb[:, :, ::-1].astype(np.uint8)
    visualization[
        : frame_rgb.shape[0] // 2, frame_rgb.shape[1] :, :
    ] = landmark_visualization[:, :, ::-1].astype(
        np.uint8
    )  # OpenCV needs BGR
    visualization[frame_rgb.shape[0] // 2 :, frame_rgb.shape[1] :, :] = (
        circumplex_bgr.astype(np.uint8)
    )

    return visualization

def extract_emotions_and_save(emonet, sfd_detector, video_dir, save_dir, start_idx, end_idx):
    image_size = 256
    device = "cuda:0"
    video_dirs = sorted(os.listdir(video_dir))[start_idx:end_idx]
    non_list = []
    for dd in tqdm(video_dirs):
        video_lists = sorted(glob.glob(os.path.join(video_dir,dd,"*.mp4")))
        # video_path = Path(__file__).parent / args.video_path

        for vv in video_lists:
            id=vv.split("/")[-2]
            utt = vv.split("/")[-1].split(".")[0]

            video_save_path = os.path.join(save_dir, id+"_"+utt+".npy")
            if os.path.isfile(video_save_path): continue
            #video_save_path = os.path.join(save_dir, vv.split("/")[-1].replace(".mp4",".npy"))

            list_frames_rgb = load_video(vv)

            vis_cnt=0
            non_cnt=0
            visualization_frames = []
            for i, frame in enumerate(list_frames_rgb):
                # Run face detector
                with torch.no_grad():
                    # Face detector requires BGR frame
                    detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])

                # If at least a face has been detected, run emotion recognition on the first face
                if len(detected_faces)>0:
                    # Only take the first detected face
                    bbox = np.array(detected_faces[0]).astype(np.int32)
                    bbox[bbox<0]=0

                    face_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

                    image_rgb = cv2.resize(face_crop, (image_size, image_size))

                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    visualization_frames.append(image_rgb)

                    temp_rgb=image_rgb
                    # emotion_prediction = run_emonet(emonet, face_crop.copy())
                    #
                    # visualization_bgr = make_visualization(
                    #     frame.copy(), face_crop.copy(), bbox, emotion_prediction
                    # )
                    # visualization_frames.append(visualization_bgr)
                else:
                    non_cnt+=1
                    # Visualization without emotion
                    # visualization = np.zeros(
                    #     (frame.shape[0], frame.shape[1] + frame.shape[0] // 2, 3),
                    #     dtype=np.uint8,
                    # )
                    # visualization[:, : frame.shape[1], :] = frame[:, :, ::-1].astype(np.uint8)
                    #
                    # visualization_frames.append(visualization)
                    visualization_frames.append(temp_rgb)

                if non_cnt>25:
                    break

                if (i+1)%100==0:
                    visualization_frames = np.array(visualization_frames)
                    image_tensor = torch.Tensor(visualization_frames).permute(0, 3, 1, 2).to(device) / 255.0
                    del visualization_frames, detected_faces

                    with torch.no_grad():
                        output = emonet(image_tensor)
                        del image_tensor
                    if vis_cnt==0:
                        final_save_frames=output.detach().cpu()
                        vis_cnt+=1
                    else:
                        final_save_frames=torch.cat((final_save_frames, output.detach().cpu()), axis=0)
                    visualization_frames=[]

                elif i==(len(list_frames_rgb)-1):

                    visualization_frames = np.array(visualization_frames)
                    image_tensor = torch.Tensor(visualization_frames).permute(0, 3, 1, 2).to(device) / 255.0
                    del visualization_frames, detected_faces

                    with torch.no_grad():
                        output = emonet(image_tensor)
                        del image_tensor
                    if i<99:
                        final_save_frames=output.detach().cpu()
                    else:
                        final_save_frames = torch.cat((final_save_frames, output.detach().cpu()), axis=0)


            # visualization_frames = np.array(visualization_frames)
            # image_tensor = torch.Tensor(visualization_frames).permute(0,3, 1, 2).to(device) / 255.0
            # del visualization_frames, list_frames_rgb, detected_faces
            # with torch.no_grad():
            #     output = emonet(image_tensor)
            #     del image_tensor
            try:
                if non_cnt>25:
                    non_list.append(vv)
                    break
                np.save(video_save_path, final_save_frames.numpy())
            except:
                non_list.append(vv)
                # pdb.set_trace()
                continue
    with open("non_list.json", "w") as f:
        json.dump(non_list, f, indent=2)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses",
        type=int,
        default=8,
        choices=[5, 8],
        help="Number of emotional classes to test the model on. Please use 5 or 8.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="video.mp4",
        help="Path to a video.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.mp4",
        help="Path where the output video is saved.",
    )

    args = parser.parse_args()

    # Parameters of the experiments

    device = "cuda:0"
    n_expression = args.nclasses
    image_size = 256
    emotion_classes = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt",
    }

    print(f"Loading emonet")
    emonet_ = load_emonet(n_expression, device)

    print(f"Loading face detector")
    sfd_detector_ = SFDDetector(device)

    print(f"Loading video")
    video_dir_ = "/scratch/10441/sk63833/datasets/voxceleb/test"
    save_dir_ = "/scratch/10441/sk63833/datasets/voxceleb_tts/xs/face_npy"

    os.makedirs(save_dir_, exist_ok=True)
    # ii=0
    # extract_emotions_and_save(emonet_, sfd_detector_, video_dir_, save_dir_, ii * 23000, (ii + 1) * 23000)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    processes = [mp.Process(target=extract_emotions_and_save, args=(emonet_, sfd_detector_, video_dir_, save_dir_, ii * 28, (ii + 1) * 28)) for ii in range(10)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # for i in range(10):  # Creating 5 processes
    #     p = mp.Process(target=extract_emotions_and_save, args=(emonet_, sfd_detector_, video_dir_, save_dir_, i * 509, (i + 1) * 509))
    #     jobs.append(p)
    #     p.start()
    # for j in jobs:
    #     j.join()
#
#
#
# import torch
# import multiprocessing as mp
#
#
# def cuda_function(queue):
#     device = torch.device('cuda')
#     tensor = torch.zeros(10).to(device)
#     queue.put(tensor.sum().item())
#
#
# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)
#     queue = mp.Queue()
#     processes = [mp.Process(target=cuda_function, args=(queue,)) for _ in range(2)]
#
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#
#     results = [queue.get() for _ in processes]
#     print("Results from CUDA processes:", results)