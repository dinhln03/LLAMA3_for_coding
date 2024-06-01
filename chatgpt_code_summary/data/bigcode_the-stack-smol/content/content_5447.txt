from torchtext import data
import spacy
import dill

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenizer_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

SRC = data.Field(tokenize=tokenizer_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenizer_en, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)

data_fields = [('German', SRC), ('English', TGT)]

train, val, test = data.TabularDataset.splits(path='./data', train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=data_fields, skip_header=True)

SRC.build_vocab(train.German)
TGT.build_vocab(train.English)

with open("./data/src_vocab.pt", "wb")as f:
    dill.dump(SRC, f)
with open("./data/tgt_vocab.pt", "wb")as f:
    dill.dump(TGT, f)
