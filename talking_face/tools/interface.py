import os
from skimage import io, img_as_float32
import cv2
import torch
import numpy as np
import subprocess
import pandas
from talking_face.talking_face.models.audio2pose import audio2poseLSTM
from scipy.io import wavfile
import python_speech_features
import pyworld
from talking_face.talking_face import config
import json
from scipy.interpolate import interp1d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inter_pitch(y, y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while i < frame_num:
        if y_flag[i] == 0:
            while True:
                if y_flag[i] == 0:
                    if i == frame_num-1:
                        if last != -1:
                            y[last+1:] = y[last]
                        i += 1
                        break
                    i += 1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i-last+1
                fy = np.array([y[last], y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx, fy)
                fx_new = np.linspace(0, 1, inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i += 1

        else:
            last = i
            i += 1
    return y


def load_ckpt(checkpoint_path, generator=None, kp_detector=None, ph2kp=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if ph2kp is not None:
        ph2kp.load_state_dict(checkpoint['ph2kp'])
    if generator is not None:
        generator.load_state_dict(checkpoint['generator'])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint['kp_detector'])


def get_img_pose(img_path):
    processor = config.OPENFACE_POSE_EXTRACTOR_PATH

    tmp_dir = config.generate_absolute_path_string("samples/tmp_dir")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        subprocess.call([processor, "-f", img_path, "-out_dir", tmp_dir, "-pose"])
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running OpenFace pose extractor: {e}")

    img_file = os.path.basename(img_path)[:-4]+".csv"
    csv_file = os.path.join(tmp_dir, img_file)
    pos_data = pandas.read_csv(csv_file)

    pose = [pos_data["pose_Rx"][0], pos_data["pose_Ry"][0], pos_data["pose_Rz"][0], pos_data["pose_Tx"][0], pos_data["pose_Ty"][0], pos_data["pose_Tz"][0]]
    pose = np.array(pose, dtype=np.float32)
    return pose


def read_img(path):
    img = io.imread(path)[:,:,:3]
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def parse_phoneme_file(phoneme_path,use_index = True):
    with open(phoneme_path,'r') as f:
        result_text = json.load(f)
    frame_num = int(result_text[-1]['phones'][-1]['ed']/100*25)
    phoneset_list = []
    index = 0

    word_len = len(result_text)
    word_index = 0
    phone_index = 0
    cur_phone_list = result_text[0]["phones"]
    phone_len = len(cur_phone_list)
    cur_end = cur_phone_list[0]["ed"]

    phone_list = []

    phoneset_list.append(cur_phone_list[0]["ph"])
    i = 0
    while i < frame_num:
        if i * 4 < cur_end:
            phone_list.append(cur_phone_list[phone_index]["ph"])
            i += 1
        else:
            phone_index += 1
            if phone_index >= phone_len:
                word_index += 1
                if word_index >= word_len:
                    phone_list.append(cur_phone_list[-1]["ph"])
                    i += 1
                else:
                    phone_index = 0
                    cur_phone_list = result_text[word_index]["phones"]
                    phone_len = len(cur_phone_list)
                    cur_end = cur_phone_list[phone_index]["ed"]
                    phoneset_list.append(cur_phone_list[phone_index]["ph"])
                    index += 1
            else:
                # print(word_index,phone_index)
                cur_end = cur_phone_list[phone_index]["ed"]
                phoneset_list.append(cur_phone_list[phone_index]["ph"])
                index += 1

    with open("phindex.json") as f:
        ph2index = json.load(f)
    if use_index:
        phone_list = [ph2index[p] for p in phone_list]
    saves = {"phone_list": phone_list}

    return saves


def get_audio_feature_from_audio(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]

    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))

    mfcc = python_speech_features.mfcc(audio, sample_rate)
    log_filterbank = python_speech_features.logfbank(audio, sample_rate)
    pitch_contour, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
    pitch_flag = (pitch_contour == 0.0) ^ 1
    pitch_contour = inter_pitch(pitch_contour, pitch_flag)
    pitch_contour = np.expand_dims(pitch_contour, axis=1)
    pitch_flag = np.expand_dims(pitch_flag, axis=1)

    frame_num = np.min([mfcc.shape[0], log_filterbank.shape[0], pitch_contour.shape[0]])
    cat = np.concatenate([mfcc[:frame_num], log_filterbank[:frame_num], pitch_contour[:frame_num], pitch_flag[:frame_num]], axis=1)
    return cat


def get_pose_from_audio(img, audio, audio2pose):

    num_frame = len(audio) // 4

    minv = np.array([-0.6, -0.6, -0.6, -128.0, -128.0, 128.0], dtype=np.float32)
    maxv = np.array([0.6, 0.6, 0.6, 128.0, 128.0, 384.0], dtype=np.float32)
    generator = audio2poseLSTM().to(device).eval()

    ckpt_para = torch.load(audio2pose, map_location=torch.device(device))

    generator.load_state_dict(ckpt_para["generator"])
    generator.eval()

    audio_seq = []
    for i in range(num_frame):
        audio_seq.append(audio[i*4:i*4+4])

    audio = torch.from_numpy(np.array(audio_seq, dtype=np.float32)).unsqueeze(0).to(device)

    poses = generator({"img": img, "audio": audio})
    poses = poses.cpu().data.numpy()[0]
    poses = (poses+1)/2*(maxv-minv)+minv

    return poses

