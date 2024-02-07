OUT_PATH = '/data/lmorove1/afavaro/data/IS_2024/transcripts_with_prompts/'
root2 = '/scratch4/lmorove1/afavaro/data/TAUKADIAL-24/TAUKADIAL-24/train/'

from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import os
token= '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'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", token))

# define a wrapper function for seeing how prompts affect transcriptions
def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
        maxBodyLength=25 * 1024 * 1024
    )
    return transcript.text

all_files_audio = []
for path, subdirs, files in os.walk(root2):
    #print(path)
    for name in files:
        if name.endswith(".wav"):
            all_files_audio.append(os.path.join(path, name))

names = []
ids = []
#index = all_files_audio.index(os.path.join(root2, 'taukdial-161-3.wav'))
for audio_file in all_files_audio[121+2:]:
        print(audio_file)
        base_name = os.path.basename(audio_file).split(".wav")[0]
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio_file,
                   prompt="Well, um, I was just, you know, walking into the kitchen, and, uh, I noticed that the cookie jar was, um, mysteriously open, and, like, there were crumbs all over the counter counter, so, ah, I think someone might might have, you know, helped themselves to a few cookies when, uh, nobody was around.")
        with open(OUT_PATH_FILE,'w') as output:
            for line in transcript:
                output.write(line)