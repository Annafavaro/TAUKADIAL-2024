from pydub import AudioSegment
import os
import subprocess
out = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_16k_no_diatization_ogg/'
input_dir = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/train_audios_original/'

#def convert_wav_to_ogg(input_file, output_file):
#    # Load the WAV file
#    sound = AudioSegment.from_wav(input_file)
#    sound = sound.set_frame_rate(16000)
#    # Export the audio in OGG format
#    sound.export(output_file, format="ogg", codec="opus")



def convert_wav_to_ogg(input_file, output_file):
    # Define the FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",  # Resample to 16 kHz
        "-ac", "2",  # Set stereo channels
        output_file
    ]

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("Conversion successful!")
    except subprocess.CalledProcessError as e:
        print("Conversion failed:", e)



all_audios = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
for audio in all_audios:
    print(audio)
    base = os.path.basename(audio).split('.wav')[0]
    out_file = os.path.join(out, base+'.ogg')
    convert_wav_to_ogg(audio, out_file)
