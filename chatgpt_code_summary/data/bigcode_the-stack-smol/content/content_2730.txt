"""Author: Brandon Trabucco
Calculate the part of speech tagger using the brown corpus.
"""

import glove.configuration
import glove.tagger


config = glove.configuration.TaggerConfiguration(
    tagger_dir="./")


glove.tagger.dump(config)


