
OUT_PATH = '/export/c06/afavaro/DementiaBank/Pitt/0extra/enhanced-cookie/transcripts_IS2024_with_prompts/'
root2 = '/export/c06/afavaro/DementiaBank/Pitt/0extra/enhanced-cookie/enhance_diarized_audios_interspeech2024/'


from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import os

#token = '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'
token = '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' #yuzhe paid
#token = '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'
##token = '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'
#token = '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'
#token= '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'
#token = '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'
#token='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'

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

# change here the -1.wav depending on the task -----> COOKIE THIEF mostly
all_files_audio = [os.path.join(root2, elem) for elem in os.listdir(root2)][378:]
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
        prompt="So, um, it is kinda like, the window is open, you know? \
        And, uh, the curtains, the curtains are pulled apart. I think the housewife is, like, probably doing the dishes or something. \
        But, um, the sink is overflowing, and she does not even seem to notice. \
        Um, the kids, they are, like, trying to reach the cookie, the cookie jar. Oh, and the boy, he is standing on this, like, wobbly three-legged stool that is about to tip over.")
        with open(OUT_PATH_FILE, 'w') as output:
            for line in transcript:
                output.write(line)
    if file_size_mb > limit_mb:
        print(f"This file is too big: {audio_file}")
        convert_to_ogg.append(audio_file)

print(convert_to_ogg)

