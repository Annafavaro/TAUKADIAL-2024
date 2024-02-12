import os
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
import math
import os
import sys
from numpy import save
import torch
import tensorflow as tf
import tensorflow_hub as hub
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import librosa

path_audios = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_16k_no_diarization/'
all_audios = [os.path.join(path_audios, elem) for elem in os.listdir(path_audios)]
output_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/wav2vec/'


def get_all_sub_segment_inds(x, fs=16e3, dur=10):
    """
        get the range of indices that can be used to run get_sub_segment()
        from the given audio signal

        - dur: number of seconds in a segment
    """
    N = x.shape[0]  # number of samples in input signal
    N_seg = dur * fs  # number of samples in a segment with the duration we want
    ind_range = math.ceil(N / N_seg)  # possible indices: 0:ind_range exclusive
    return ind_range


def get_sub_segment(x, fs=16e3, dur=10, index=0):
    """
        Get a segment of the input signal x

        - dur: number of seconds in a segment
        - index: index of the segment counted from the whole signal
    """
    # check if segment out of input range
    N = x.shape[0]  # number of samples in input signal
    start_pt = int(index * dur * fs)
    end_pt = int(start_pt + dur * fs)
    if end_pt > N:
        end_pt = N

    # get segment
    seg = x[start_pt:end_pt]
    # zero padding at the end to dur if needed
    if seg.shape[0] < (dur * fs):
        pad_len = int((dur * fs) - seg.shape[0])
        seg = np.pad(seg, ((0, pad_len)), 'constant')

    return seg

model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

for audio in all_audios:
    print(audio)
    base = os.path.basename(audio).split('.wav')[0]
    x, fs = librosa.load(audio, sr=16000)
    x = x / (max(abs(x)))
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10) # 10sec segments
    embeddings = np.zeros(shape=(ind_range, 512))

    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        inputs = feature_extractor(seg, sampling_rate=16000, padding=True, return_tensors="pt")
        hidden_states = model(inputs.input_values)
        embeddings[spec_ind,:] = hidden_states['extract_features'].squeeze().mean(dim=0).detach().numpy()
        # This gives you the sequence of features from the last convolutional layer of the model.

    hidden_layer_avg = np.mean(embeddings, axis=0)
    hidden_layer_avg = hidden_layer_avg.reshape((1, 512))
    out_path = os.path.join(output_path, base + '.npy')
    save(out_path, hidden_layer_avg)