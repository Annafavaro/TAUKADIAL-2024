from transformers import AutoFeatureExtractor, WavLMForXVector
import torch
import sys
import os
import librosa

if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    model_name = "microsoft/wavlm-base-plus-sv"

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    all_audios = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]

    for audio in all_audios:
        print(audio)
        base = os.path.basename(audio).split('.wav')[0]
        x, fs = librosa.load(audio, sr=16000)
        inputs = feature_extractor(x, sampling_rate=16000, padding=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        print(type(embeddings))
        print(embeddings.shape)

