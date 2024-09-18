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

# change here the -1.wav depending on the task --> driving car
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2) if '.wav' in elem  and '-1.wav' in elem]

convert_to_ogg = []

for audio_file in all_files_audio:
    print(audio_file)
    file_size_bytes = os.path.getsize(audio_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= limit_mb:
        base_name = os.path.basename(audio_file).split(".wav")[0]
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio_file,
        prompt="So, um, there is this group of people, right? \
        And, uh, they are all, like, driving in a car. \
        In the lower picture, everyone seems pretty chill, just, you know, looking out the window. There is this boy and a girl, a girl looking out, and, uh, there is a dog, \
        and a man driving, and, um, a woman, a woman who looks like she is asleep with a kid next to her in the car. \
        And there are, um, two people in the back seat, probably a grandma and a kid.")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
    if file_size_mb > limit_mb:
        print(f"This file is too big: {audio_file}")
        convert_to_ogg.append(audio_file)

print(convert_to_ogg)


