from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils.midifile import *
from musicautobot.utils.file_processing import process_all

from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils.midifile import *
from musicautobot.utils.file_processing import process_all

import random

def create_databunch(files, data_save_name, path):
    save_file = path/data_save_name
    if save_file.exists():
        data = load_data(path, data_save_name)
    else:
        save_file.parent.mkdir(exist_ok=True, parents=True)
        vocab = MusicVocab.create()
        processors = [OpenNPFileProcessor(), MusicItemProcessor()]

        data = MusicDataBunch.from_files(files, path, processors=processors, encode_position=True)
        data.save(data_save_name)
    return data

def timeout_func(data, seconds):
    print("Timeout:", seconds)

def process_metadata(midi_file):
    # Get outfile and check if it exists
    out_file = numpy_path/midi_file.relative_to(midi_path).with_suffix('.npy')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists(): return
    
    npenc = transform_midi(midi_file)
    if npenc is not None: np.save(out_file, npenc)

def transform_midi(midi_file):
    input_path = midi_file
    
    # Part 1: Filter out midi tracks (drums, repetitive instruments, etc.)
    try: 
#         if duet_only and num_piano_tracks(input_path) not in [1, 2]: return None
        input_file = compress_midi_file(input_path, min_variation=min_variation, cutoff=cutoff) # remove non note tracks and standardize instruments
        
        if input_file is None: return None
    except Exception as e:
        if 'badly form' in str(e): return None # ignore badly formatted midi errors
        if 'out of range' in str(e): return None # ignore badly formatted midi errors
        print('Error parsing midi', input_path, e)
        return None
        
    # Part 2. Compress rests and long notes
    stream = file2stream(input_file) # 1.
    try:
        chordarr = stream2chordarr(stream) # 2. max_dur = quarter_len * sample_freq (4). 128 = 8 bars
    except Exception as e:
        print('Could not encode to chordarr:', input_path, e)
        print(traceback.format_exc())
        return None
    
    # Part 3. Compress song rests - Don't want songs with really long pauses 
    # (this happens because we filter out midi tracks).
    chord_trim = trim_chordarr_rests(chordarr)
    chord_short = shorten_chordarr_rests(chord_trim)
    delta_trim = chord_trim.shape[0] - chord_short.shape[0]
#     if delta_trim > 500: 
#         print(f'Removed {delta_trim} rests from {input_path}. Skipping song')
#         return None
    chordarr = chord_short
    
    # Part 3. Chord array to numpy
    npenc = chordarr2npenc(chordarr)
    if not is_valid_npenc(npenc, input_path=input_path):
        return None
    
    return npenc

# Location of your midi files
midi_path = Path('data/midi/TPD')
# Location of preprocessed numpy files
numpy_path = Path('data/numpy/preprocessed data')

# Location of models and cached dataset
data_path = Path('data/cached')
data_save_name = 'TPD_musicitem_data_save.pkl'

# num_tracks = [1, 2] # number of tracks to support
cutoff = 5 # max instruments
min_variation = 3 # minimum number of different midi notes played
# max_dur = 128

midi_files = get_files(midi_path, '.mid', recurse=True)

print('Loading model...')
batch_size = 1
encode_position = True
dl_tfms = [batch_position_tfm] if encode_position else []
data = load_data(data_path, data_save_name, bs=batch_size, encode_position=encode_position, dl_tfms=dl_tfms)

config = default_config()
config['encode_position'] = encode_position
learn = music_model_learner(data, config=config.copy())

learn.fit_one_cycle(4)
learn.save('TPD_model')
