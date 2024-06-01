import song_generator
from  markov_gen import markov_generator
import os, json


nino_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

import generator
generator.set_dir_write_note(nino_dir + '/trainer/generated_notes')


def gen_kwargs():
    kwargs = {
        #What the general scale for the shond should be - chosen randomly from this list. 
        'song_scale' : ['major', 'minor'],

        'block_same_note_range' : [0], 

        #How many segments the song has. 
        'number_of_segments_range' : [1],

        'chord_exp_var': 2,

    
        #The range of BPMs for each segment. Chooses randomly for each segment. 
        'bpm_range': [400], 

        #A range for beats per bar for each segment. Will choose randomly. 
        'beats_per_bar_range' : [4],

        #A range for how many chords each segment should have. Chooses randomly from this list. 
        'chords_per_segment_range': [1], 

        #A list containing the program numbers of instruments that should be used. The program will choose randomly from these. 
        'instruments_range' : [1, 2],

        #The number of instruments that will be active throughout the song. 
        'number_of_song_instruments_range' : [1],

        #The bias_same_note chances for the main instruments. It's a list that is randomly chosen from. 
        'main_instrument_bias_same_note' : [0],

        #The maximum pattern note length for the main instrument
        'pattern_note_max_len_range' : 1,

        #Each segment will have a different accent from the previous, determined by a random value from this list. 
        'accent_offset' : range(-5, 5),

        #A list from accents from which segments will receive their base accent. 
        'default_accent_range' : range(70, 85),

        #Volume to be added to percussions. 
        'percussion_accent_offset' : 10,

         #Number of extra instruments per segment. 
        'no_segment_instruments_range' : [0],

        #Range for the number of bars per segment. Will choose randomly from this list. 
        'number_segment_bars_range' : [32],

        #Accent range offset for instrument specific blocks. 
        'block_default_accent_range' :  range(-5, 5),

        #Chance for each instrument to follow a pattern for the duration of the segment. 
        'segment_instrument_pattern_chance' : 1.0,

        #Upper range for how long a pattern note can last. Should not be longer than the maximum amount of beats per bar for the instrument. 
        'pattern_note_len_range' :  1,
        
        #And the lower range for how long the note can last. Should not be less than 1. 
        'pattern_note_min_len_range' : 1,

        #The dir where the songs are saved. 
        'generate_dir' :  nino_dir + '/trainer/',

        #The directory for the soundfont. This is an example, and should be supplied for specific use cases. 
        'soundfont' :  nino_dir + '/soundfonts/FluidR3_GM.sf2',

        #The song generator will randomly repeat segments and then shuffle them. This is a range of the numbers of repeats for each segment. 
        'segment_shuffle_range' : [1],

        #We may want to have segments with few instruments and no drums. This is the percentage that there are drums if the number of instruments is below the defined treshold. 
        'segment_percussion_chance': 0.0,

        #If there's less than this many instruments in a segment, there's a chance (defined above) that there will be no percussion for that segment. 
        'skip_percussion_treshold' : 3,

        'get_mp3' : True,
        
        'dir_write_note' : nino_dir + '/trainer/generated_notes',
        'markov_values' : nino_dir + '/trainer/results.json',
    }
    return kwargs


def main():
    kwargs = gen_kwargs()
    song_generator.generate_song(**kwargs)

    markov_weight = float(raw_input('Enter a weight for the markov values. '))

    generated_notes = ''
    with open(nino_dir + '/trainer/generated_notes') as f: 
        generated_notes = '[' + f.read()[2:] + ']'

    generated_notes = json.loads(generated_notes)

    old_results = ''
    with open(nino_dir + '/trainer/results.json') as f: 
        old_results = json.loads(f.read())

    new_results = markov_generator.markov_from_values(old_results, generated_notes, 4, weight = markov_weight)
    with open(nino_dir + '/trainer/results.json', 'w') as f: 
        f.write(json.dumps(new_results))


if __name__ == '__main__' : 
    main()
