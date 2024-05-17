#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import h5py
import random
import librosa
import tqdm

# 设置随机数生成器的种子，以确保结果可重复
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('wav_csv', type=str)
parser.add_argument('feature_h5', type=str)
parser.add_argument('--feature_csv', type=str, default=None)
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument('--sr', type=int, default=44100)
parser.add_argument('--winlen', default=40, type=float, help='FFT duration in ms')
parser.add_argument('--hoplen', default=20, type=float, help='hop duration in ms')
parser.add_argument('--n_mels', default=64, type=int)
args = parser.parse_args()

if args.feature_csv is None:
    args.feature_csv = Path(args.feature_h5).with_suffix(".csv")

wav_df = pd.read_csv(args.wav_csv, sep='\\s+')

mel_args = {
    'n_mels': args.n_mels,
    'n_fft': 2048,
    'hop_length': int(args.sr * args.hoplen / 1000),
    'win_length': int(args.sr * args.winlen / 1000),
    'sr': args.sr
}

def augment_audio(y, sr):
    # 噪声注入
    noise_amp = 0.005 * random.uniform(0, 1)
    noise = noise_amp * np.random.randn(*y.shape)
    y_aug = y + noise

    # 时间平移
    shift_samples = int(sr * random.uniform(-0.1, 0.1))
    if shift_samples > 0:
        y_aug = np.pad(y, (0, shift_samples), 'constant')[:-shift_samples]
    else:
        y_aug = np.pad(y, (-shift_samples, 0), 'constant')[-shift_samples:]

    # 音调变化
    pitch_shift_samples = int(random.uniform(-1, 1) * sr / 2)  # 半音变化
    y_aug = librosa.effects.pitch_shift(y_aug, sr, n_steps=pitch_shift_samples)

    # 时间伸缩
    speed_change = random.uniform(0.8, 1.2)
    y_aug = librosa.effects.time_stretch(y_aug, speed_change)

    # 确保音频长度与原始信号一致
    y_fixed = librosa.util.fix_length(y_aug, y.shape[0])

    return y_fixed

def extract_feature(item):
    row = item[1]
    y, sr = sf.read(row["file_name"], dtype='float32')
    if y.ndim > 1:
        y = y.mean(1)
    
    # 应用数据增强
    y_aug = augment_audio(y, sr)
    
    # 重新采样，然后提取梅尔频谱特征
    y_aug = librosa.resample(y_aug, orig_sr=sr, target_sr=args.sr)
    mel_spec = librosa.feature.melspectrogram(y=y_aug, **mel_args)
    lms = librosa.power_to_db(mel_spec, top_db=None).T
    
    return row['audio_id'], lms

feat_csv_data = []
store_path = Path(args.feature_h5).absolute().__str__()
with h5py.File(args.feature_h5, 'w') as store, tqdm(total=wav_df.shape[0]) as pbar:
    for aid, feat in pr.map(extract_feature, wav_df.iterrows(),
                              workers=args.num_worker, maxsize=4):
        store[aid] = feat
        feat_csv_data.append({
            "audio_id": aid,
            "hdf5_path": store_path
        })
        pbar.update()

pd.DataFrame(feat_csv_data).to_csv(args.feature_csv, sep="\t", index=False)