import torch
import os
import sys
from numpy import save

if __name__ == "__main__":

    sound_dir = sys.argv[1]
    output_dir = sys.argv[2]

    all_audios = sorted([os.path.join(sound_dir, elem) for elem in os.listdir(sound_dir)])
    model_vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model_vggish.eval()

    for audio_file in all_audios:
        print(f"Vggish audio {audio_file}...")
        base = os.path.basename(audio_file).split('.wav')[0]
        features_vggish = model_vggish.forward(audio_file)
        features_vggish_np = features_vggish.detach().numpy() # Convert to numpy
        print(features_vggish_np.shape)
        print(type(features_vggish_np))
        save(output_dir + base + '.npy', features_vggish_np)