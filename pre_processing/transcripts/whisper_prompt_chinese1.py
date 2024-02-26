
#OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/transcripts_prompts_refined/chinese/'
OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/transcripts_with_prompt/chinese/'
#root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/audios/chinese/'
root2 = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/data_divided_by_language/audios/chinese/'

from openai import OpenAI  # for making OpenAI API calls
import os

token = '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'
#token = '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' #yuzhe paid
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
        language='zh'
    )
    return transcript.text

#set_of_sps = ['taukdial-064-1', 'taukdial-004-1', 'taukdial-136-1', 'taukdial-164-1' ]
# change here the -1.wav depending on the task
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2) if '.wav' in elem  and '-1.wav' in elem]
print(len(all_files_audio))

convert_to_ogg = []

for audio_file in all_files_audio:
    print(audio_file)
    file_size_bytes = os.path.getsize(audio_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= limit_mb:
        base_name = os.path.basename(audio_file).split(".wav")[0]
       # if base_name in set_of_sps:
            print(f'yes----> {base_name}')
            OUT_PATH_FILE = os.path.join(OUT_PATH, base_name + '.txt')
            transcript = transcribe(audio_file,
            prompt="在一个公共场所夜市,有一个摊位是掷骰子的，然后呢，有很多小朋友，嗯，有几个小朋友在掷骰子吧，很专注地在玩，啊，没想到旁边有一个小偷，趴手，\
            他趁小孩子不留意的时候伸手到小孩子的背包里面，可能想偷取一些值钱的东西。另外呢，有妈妈带着小朋友一个捞鱼的池子旁边捞鱼。 \
            妈妈在指着鱼和妹妹说话,但是她太专注了，就没有想到自己手上拿着冰淇淋滴滴滴滴滴到妹妹头发上，把头发滴湿了，妹妹没有察觉，还吐着舌头，自得其乐。 \
            还有个奶奶抱着孙子，搭着孙子，一脸幸福的微笑，孙子玩得很开心，他捞到了鱼了，露出可爱的笑容。 另外一个应该是姐姐，她没有捞到鱼,有一点点失望。 哦，这边那个小贩，\
             嗯，他有烤香肠，烤香肠应该是香喷喷的吧，他们一个只顾捞鱼玩，一个只顾掷彩子玩，连小偷偷到背包他们都没有察觉。")
            with open(OUT_PATH_FILE, 'w') as output:
                for line in transcript:
                    output.write(line)
    if file_size_mb > limit_mb:
        print(f"This file is too big: {audio_file}")
        convert_to_ogg.append(audio_file)

print(convert_to_ogg)


