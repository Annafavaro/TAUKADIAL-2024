OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/transcripts_prompts_refined/english/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/audios/english/'

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
        language='en'
    )
    return transcript.text

# change here the -1.wav depending on the task
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2) if '.wav' in elem  and '-1.wav' in elem]

convert_to_ogg = []

for audio_file in all_files_audio:
    file_size_bytes = os.path.getsize(audio_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= limit_mb:
        base_name = os.path.basename(audio_file).split(".ogg")[0]
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio_file,
        prompt="Well, um, I was just, you know, walking into the kitchen,"
               "and, uh, I noticed that the cookie jar was, um, mysteriously open, and, like,"
               "there were crumbs all over the counter counter, so, ah, I think someone might might have,"
               "you know, helped themselves to a few cookies when, uh, nobody was around.")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
    if file_size_mb > limit_mb:
        convert_to_ogg.append(audio_file)

    print(convert_to_ogg)






  # # audio_file_complete = os.path.join(root2, audio +'.wav')
  #  base_name = os.path.basename(audio).split(".ogg")[0]
  #  OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
  #  transcript = transcribe(audio,
  #  prompt="Well, um, I was just, you know, walking into the kitchen, and, uh, I noticed that the cookie jar was, um, mysteriously open, and, like, there were crumbs all over the counter counter, so, ah, I think someone might might have, you know, helped themselves to a few cookies when, uh, nobody was around.")
  #  with open(OUT_PATH_FILE, 'w') as output:
  #      for line in transcript:
  #          output.write(line)
#f#or audio_file in all_files_audio[377:]:
  #      #print(audio_file)
  #     # file_size_bytes = os.path.getsize(audio_file)
  #      #file_size_mb = file_size_bytes / (1024 * 1024)
  #     # if file_size_mb <= limit_mb:
#