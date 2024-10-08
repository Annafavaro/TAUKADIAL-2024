# conda activate openai

#OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/transcripts_prompts_refined/chinese/'
OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/transcripts_with_prompt/english/'
#root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/audios/chinese/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/audios/english/'
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
        prompt=prompt,
        language='en'
    )
    return transcript.text

# change here the -2.wav depending on the task --> ladder
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2) if '.wav' in elem  and '-2.wav' in elem]
print(len(all_files_audio))
convert_to_ogg = []

for audio_file in all_files_audio:
    print(audio_file)
    file_size_bytes = os.path.getsize(audio_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= limit_mb:
        base_name = os.path.basename(audio_file).split(".wav")[0]
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio_file,
        prompt="So, it is kinda like, this little girl is cat, you know, climbed up the tree tree and got stuck there, right? \
               And then she told her dad about it, and he, um, went and grabbed a ladder, a ladder, and he, like, climbed up the tree, but then, uh, the ladder fell down, \
              and now he is stuck up there with the cat. And, uh, the dog, the dog is down at the bottom, barking for him, I guess.")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
    if file_size_mb > limit_mb:
        print(f"This file is too big: {audio_file}")
        convert_to_ogg.append(audio_file)

print(convert_to_ogg)

