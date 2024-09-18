OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/temp/'
#root2 = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_16k_no_diarization/'

from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import os

token = '<YOUR_TOKEN>'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", token))
limit_mb=25

# define a wrapper function for seeing how prompts affect transcriptions
def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
       # language ='zh',
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

names = ['taukdial-161-1', 'taukdial-132-3', 'taukdial-107-3', 'taukdial-105-1']

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
