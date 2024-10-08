#OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/transcripts_prompts_refined/chinese/'
OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/transcripts_with_prompt/chinese/'
#root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/audios/chinese/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/audios/chinese/'

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
        language='zh'
    )
    return transcript.text

set_of_sps = ['taukdial-003-2', 'taukdial-004-2', 'taukdial-110-2' ]
# change here the -1.wav depending on the task
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2) if '.wav' in elem  and '-2.wav' in elem]

print(len(all_files_audio))
convert_to_ogg = []

for audio_file in all_files_audio:
    print(audio_file)
    file_size_bytes = os.path.getsize(audio_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= limit_mb:
        base_name = os.path.basename(audio_file).split(".wav")[0]
      #  if base_name in set_of_sps:
        print(f'yes----> {base_name}')
        OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
        transcript = transcribe(audio_file,
        prompt="这是在一个公园，两个女孩在打网球或者羽毛球，那，有一个人牵着一只狗，还有那个小狗，小狗就很调皮的抓着那个打那个羽毛球的小朋友，咬住其中一个女孩的衣服，\
        那个女孩是一脸惊恐。另外，有两个老人家下，下棋，下象棋，然后就很开心。旁边还有泡的茶，茶还冒着烟。另外两个年纪更小的小朋友在稍远的院子里，\
        在溜滑梯荡秋千，也很开心。")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
    if file_size_mb > limit_mb:
        print(f"This file is too big: {audio_file}")
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