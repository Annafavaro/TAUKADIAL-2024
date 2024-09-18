OUT_PATH = '/data/lmorove1/afavaro/data/IS_2024/transcripts_with_prompts/'
root2 = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train/'
OUT_PATH2 = '/data/lmorove1/afavaro/data/IS_2024/audios_compressed/'
target_size_mb = 25

import os
from pydub import AudioSegment
def compress_audio(input_file, output_file, target_size_mb):
    audio = AudioSegment.from_file(input_file)
    target_size_bytes = target_size_mb * 1024 * 1024  # convert MB to bytes
    audio.export(output_file, format="wav", bitrate=f"{target_size_bytes * 8 // audio.duration_seconds:.0f}k")

all_files_audio = []
for path, subdirs, files in os.walk(root2):
    #print(path)
    for name in files:
        if name.endswith(".wav"):
            all_files_audio.append(os.path.join(path, name))
names = []
ids = []
base_audios = [os.path.basename(audio).split('.wav')[0] for audio in all_files_audio]
present_tr = [os.path.basename(tr).split('.txt')[0] for tr in os.listdir(OUT_PATH)]

to_do_list = list(set(base_audios)^set(present_tr))

for audio in to_do_list:
    audio_file_complete = os.path.join(root2, audio +'.wav')
    base_name = os.path.basename(audio_file_complete).split(".wav")[0]
    print(base_name)
    #OUT_PATH_FILE = os.path.join(OUT_PATH2, base_name + '.wav')
   # compress_audio(audio_file_complete, OUT_PATH_FILE, target_size_mb)
