OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/temp/'
#root2 = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_16k_no_diarization/'

from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import os
#token = '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'
token = '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' #yuzhe paid
#token = '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'
#token = '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'
#token = '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'
#token= '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'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", token))
limit_mb=25

# define a wrapper function for seeing how prompts affect transcriptions
def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript.text

all_files_audio = []
for path, subdirs, files in os.walk(root2):
    print(path)
    for name in files:
        if name.endswith(".wav"):
            all_files_audio.append(os.path.join(path, name))

names = []
ids = []

#base_audios = [os.path.basename(audio).split('.wav')[0] for audio in all_files_audio]
#present_tr = [os.path.basename(tr).split('.txt')[0] for tr in os.listdir(OUT_PATH)]
#
#to_do_list = list(set(base_audios)^set(present_tr))


names = ['taukdial-003-3', 'taukdial-006-2', 'taukdial-006-3', 'taukdial-007-1',
'taukdial-007-2', 'taukdial-007-3', 'taukdial-023-3', 'taukdial-024-2',
'taukdial-039-2', 'taukdial-048-3', 'taukdial-065-1', 'taukdial-072-1',
'taukdial-072-3', 'taukdial-080-3', 'taukdial-093-3', 'taukdial-097-1',
'taukdial-097-2', 'taukdial-097-3', 'taukdial-109-1', 'taukdial-110-1',
'taukdial-110-2', 'taukdial-111-2', 'taukdial-131-1', 'taukdial-131-2',
'taukdial-136-2', 'taukdial-141-2', 'taukdial-157-1', 'taukdial-157-2',
'taukdial-157-3', 'taukdial-159-1', 'taukdial-159-2', 'taukdial-159-3']


for audio in all_files_audio:
   # audio_file_complete = os.path.join(root2, audio +'.wav')
    base_name = os.path.basename(audio).split(".wav")[0]
    if base_name in names:
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio,
                                prompt="Well, um, I was just, you know, walking into the kitchen, and, uh, I noticed that the cookie jar was, um, mysteriously open, and, like, there were crumbs all over the counter counter, so, ah, I think someone might might have, you know, helped themselves to a few cookies when, uh, nobody was around.")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
#for audio_file in all_files_audio[377:]:
        #print(audio_file)
       # file_size_bytes = os.path.getsize(audio_file)
        #file_size_mb = file_size_bytes / (1024 * 1024)
       # if file_size_mb <= limit_mb:
