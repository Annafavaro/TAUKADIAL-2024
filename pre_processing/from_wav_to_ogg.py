from pydub import AudioSegment
import os

out = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_16k_no_diatization_ogg/'
input_dir = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_original/'

def convert_wav_to_ogg(input_file, output_file):
    # Load the WAV file
    sound = AudioSegment.from_wav(input_file)
    sound = sound.set_frame_rate(16000)
    # Export the audio in OGG format
    sound.export(output_file, format="ogg", codec="libvorbis")

all_audios = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
for audio in all_audios:
    print(audio)
    base = os.path.basename(audio)
    out_file = os.path.join(out, base)
    convert_wav_to_ogg(audio, out_file)
