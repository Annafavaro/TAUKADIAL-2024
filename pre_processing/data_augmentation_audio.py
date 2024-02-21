import os
import librosa
from audiomentations import AddBackgroundNoise, PolarityInversion
import sys
import soundfile as sf
from audiomentations import TimeMask

def add_background_music(input_dir, output_path):

    files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
    for filename in files:
        print(filename)
        basename = os.path.basename(filename)
        y, sr = librosa.load(filename)
        transform = AddBackgroundNoise(
            # sounds_path="/export/b01/afavaro/datasets/ESC-50-master/audio/",
            sounds_path = '/export/corpora5/JHU/musan/music/',
            min_snr_in_db = 10.0,
            max_snr_in_db = 30.0,
            noise_transform=PolarityInversion(),
            p=1.0)
        augmented_sound = transform(y, sample_rate=sr)
        out_path_final = os.path.join(output_path, basename)
        sf.write(f'{out_path_final}_music.wav', augmented_sound, sr)


def add_background_noise(input_dir, output_path):

    files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
    for filename in files:
        print(filename)
        basename = os.path.basename(filename)
        y, sr = librosa.load(filename)
        transform = AddBackgroundNoise(
            # sounds_path="/export/b01/afavaro/datasets/ESC-50-master/audio/",
            sounds_path='/export/corpora5/JHU/musan/noise/',
            min_snr_in_db = 10.0,
            max_snr_in_db = 30.0,
            noise_transform=PolarityInversion(),
            p=1.0)

        augmented_sound = transform(y, sample_rate=sr)
        out_path_final = os.path.join(output_path, basename)
        sf.write(f'{out_path_final}_noise.wav', augmented_sound, sr)

def add_background_speech(input_dir, output_path):

    files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
    for filename in files:
        print(filename)
        basename = os.path.basename(filename)
        y, sr = librosa.load(filename)
        transform = AddBackgroundNoise(
            # sounds_path="/export/b01/afavaro/datasets/ESC-50-master/audio/",
            sounds_path = '/export/corpora5/JHU/musan/speech/',
            min_snr_in_db = 10.0,
            max_snr_in_db = 30.0,
            noise_transform = PolarityInversion(), p=1.0)
        augmented_sound = transform(y, sample_rate=sr)
        out_path_final = os.path.join(output_path, basename)
        sf.write(f'{out_path_final}_speech.wav', augmented_sound, sr)


def add_timemasking(input_dir, output_path):

    files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
    for filename in files:
        print(filename)
        basename = os.path.basename(filename)
        y, sr = librosa.load(filename)
        transform = TimeMask(
            min_band_part=0.15,
            max_band_part=0.25,
            fade=True,
            p=1.0,
        )
        augmented_sound = transform(y, sample_rate=sr)
        out_path_final = os.path.join(output_path, basename)
        sf.write(f'{out_path_final}_timemask.wav', augmented_sound, sr)





if __name__ == "__main__":

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # files = get_files()
    print("adding music")
    add_background_music(input_dir, output_dir)
    print("adding noise")
    add_background_noise(input_dir, output_dir)
    print("adding speech")
    add_background_speech(input_dir, output_dir)
    print("adding time masking")
    add_timemasking(input_dir, output_dir)

