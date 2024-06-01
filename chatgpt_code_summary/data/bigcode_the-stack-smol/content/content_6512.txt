import time
import scipy.io.wavfile as wavfile
import numpy as np
import speech_recognition as sr
import librosa
import argparse
import os
from glob import glob

from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from pydub.playback import play
import pysrt
import math
import shutil


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to video *.mp4 file')
    parser.add_argument('-o', '--output', type=str, default='output/', help='Output file location')
    parser.add_argument('-l', '--lang', type=str, default='en', help='Language of the video file')
    arguments = parser.parse_args()
    return arguments

def recognize(wav_filename, lang):
    data, s = librosa.load(wav_filename)
    librosa.output.write_wav('output/tmp.wav', data, s)
    y = (np.iinfo(np.int32).max * (data/np.abs(data).max())).astype(np.int32)
    wavfile.write('output/tmp_32.wav', s, y)

    r = sr.Recognizer()
    with sr.AudioFile('output/tmp_32.wav') as source:
        audio = r.record(source)  

    print('Audio file has been loaded')

    try:
        result = r.recognize_google(audio, language = lang).lower()
    except sr.UnknownValueError:
        print("Failed to determine audio file")
        result = ''
    # finally:    
    #     os.remove(wav_filename)  

    return result

def get_audio(videofile, audiofile):
    os.system('ffmpeg -y  -threads 4 -i {} -f wav -ab 192000 -vn {}'.format(videofile, audiofile))

def split_into_frames(audiofile, samplesLocation):
    os.system('rm {}/*'.format(samplesLocation))
    time.sleep(2.0)
    data, sr = librosa.load(audiofile)
    duration = librosa.get_duration(data, sr)
    print('video duration, hours: {}'.format(duration/3600))
    for i in range(0,int(duration-1),20):
        tmp_batch = data[(i)*sr:sr*(i+20)]
        librosa.output.write_wav('{}/{}.wav'.format(samplesLocation, chr(int(i/20)+65)), tmp_batch, sr)

def separate_music_voice(audioFile, outputLocation):
    os.system('spleeter separate -i {} -p spleeter:2stems -o {}'.format(audioFile, outputLocation))


# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def get_timestamp(duration):
    hr = math.floor(duration / 3600000)
    total_min = duration % 3600000
    
    mins = math.floor(total_min / 60000)
    total_secs = total_min % 60000

    secs = math.floor(total_secs / 1000)
    milisecs = total_min % 1000

    return "{:02d}:{:02d}:{:02d},{:03d}".format(hr, mins, secs, milisecs)


def gen_subtitle(wavFile, samplesLocation, srtFile, lang):
    srt_file = pysrt.SubRipFile()

    # Load your audio.
    print("loading wav file...")
    # song = AudioSegment.from_mp3("your_audio.mp3")
    #song = AudioSegment.from_wav("vocals.wav")
    song = AudioSegment.from_file(wavFile, format="wav")
    # play(song)
    dBFS = song.dBFS


    # Nonsilence track start and end positions.
    nonsilence = detect_nonsilent(
        song,
        min_silence_len = 500,
        silence_thresh = dBFS-16
    )
    file_count = len(nonsilence)
    print("Nonsilence chunk length {}".format(str(file_count)))

    # for [start, end] in nonsilence:
    #     print("start: {0} end: {1}".format(get_timestamp(start), get_timestamp(end)))

    # Split track where the silence is 2 seconds or more and get chunks using 
    # the imported function.
    print("Start spliting file...")
    chunks = split_on_silence(
        song, 
        min_silence_len = 500,
        silence_thresh = dBFS-16,
        # optional
        keep_silence = 250
    )

    print("Spliting done..." + str(len(chunks)))
    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=1000)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk
        # audio_chunk = chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        # Export the audio chunk with new bitrate.
        starttime = nonsilence[i][0]
        endtime = nonsilence[i][1]
        print("\n>>{} of {}, Exporting {}chunk{}.wav start: {} end: {}".format(i, file_count, samplesLocation, i, starttime, endtime))

        chunk_file_path = "{}chunk{}.wav".format(samplesLocation, str(i))
        normalized_chunk.export(
            chunk_file_path,
            bitrate = "192k",
            format = "wav"
        )
        
        time.sleep(2)
        print("Going to generete the dialogs of file {}".format(chunk_file_path))
        dialogs = recognize(chunk_file_path, lang)
        print("{} file dialog is: {}".format(chunk_file_path, dialogs))
        
        start_time = get_timestamp(starttime)
        end_time = get_timestamp(endtime)
        sub = pysrt.SubRipItem((i+1), start=start_time, end=end_time, text="{} {}".format(str(i+1), dialogs))
        srt_file.append(sub)

    srt_file.save(srtFile)


if __name__ == '__main__':
    outputLoc = 'output/'
    inputWaveFile = 'current.wav'
    vocals_file = 'current/vocals.wav'
    samples_location = 'samples/'
    srt_file = '.srt'

    start = time.time()

    args = get_arguments()

    outputLoc = args.output
    shutil.rmtree(outputLoc)
    time.sleep(2)
    os.makedirs(outputLoc, exist_ok=True)
    inputWaveFile = outputLoc + inputWaveFile
    vocals_file = outputLoc + vocals_file
    samples_location = outputLoc + samples_location
    os.makedirs(samples_location, exist_ok=True)
    srt_file = os.path.splitext(args.video)[0] + srt_file
    print('srt file will be {}'.format(srt_file))
    time.sleep(2)

    get_audio(args.video, inputWaveFile)
    separate_music_voice(inputWaveFile, outputLoc)
    gen_subtitle(vocals_file, samples_location, srt_file, args.lang)
    
    end = time.time()
    print('elapsed time: {}'.format(end - start))
    # shutil.rmtree(outputLoc)
