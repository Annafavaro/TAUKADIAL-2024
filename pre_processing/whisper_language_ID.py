OUT_PATH = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/lang_id_train/'
import json
import os
import pandas as pd
import whisperx

YOUR_HF_TOKEN = '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'
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

root2 = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train/'

all_files_audio = []
for path, subdirs, files in os.walk(root2):
    #print(path)
    for name in files:
        if name.endswith(".wav"):
            all_files_audio.append(os.path.join(path, name))

#index = all_files_audio.index(os.path.join(root2, 'taukdial-161-3.wav'))

names = []
ids = []

for audio_file in all_files_audio:

        base_name = os.path.basename(audio_file).split(".wav")[0]
        print(base_name)
        names.append(base_name)
        csv_path = os.path.join(OUT_PATH, base_name + ".csv")
        print(csv_path)
        json_path = os.path.join(OUT_PATH, base_name + ".json")
        print(json_path)

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        if 'taukdial-004-1' in audio_file or 'taukdial-110-2' in audio_file or 'taukdial-161-3' in audio_file:
            result_lang = 'zh'
        else:
            result_lang = str(result["language"])
        ids.append(result_lang)

dict = {'names': names, 'lang': ids}

df = pd.DataFrame(dict)
df.to_csv(os.path.join(OUT_PATH, 'lang_ids.csv'))

