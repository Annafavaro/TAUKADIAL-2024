OUT_PATH = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train_transcriptions_and_diarization/'

import json
import os
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

index = all_files_audio.index(os.path.join(root2, 'taukdial-004-1.wav'))

for audio_file in all_files_audio[index:]:

        base_name = os.path.basename(audio_file).split(".wav")[0]
        print(base_name)

        csv_path = os.path.join(OUT_PATH, base_name + ".csv")
        print(csv_path)
        json_path = os.path.join(OUT_PATH, base_name + ".json")
        print(json_path)

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment

        # 2. Align whisper output
        #model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
        if 'taukdial-004-1'  in audio_file:
            model_a, metadata = whisperx.load_align_model(language_code='zh', device=device)
        else:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio_file)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        ##print(diarize_segments)
        # print(result["segments"]) #
        diarize_segments.to_csv(csv_path)
        with open(json_path, "w") as outfile:
            json.dump(result, outfile)