from data import *
from model import *
from utils import *

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import numpy as np


#import io
#import torchvision
#from PIL import Image
#import visdom
#vis = visdom.Visdom()

def evaluate(input_seq, encoder, decoder, max_length=MAX_LENGTH):
    # input_lengths = [len(input_seq)] xiba, 嚴重錯誤
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_lengths = [len(x) for x in input_seqs]
    
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    
    if USE_CUDA:
        input_batches = input_batches.cuda()
        
    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention[0][0].cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train()
    decoder.train()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

#def show_plot_visdom():
    #buf = io.BytesIO()
    #plt.savefig(buf)
    #buf.seek(0)
    #attn_win = 'attention (%s)' % hostname
    #vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

#def show_attention(input_sentence, output_words, attentions):
    ## Set up figure with colorbar
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(attentions.numpy(), cmap='bone')
    #fig.colorbar(cax)

    ## Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    #ax.set_yticklabels([''] + output_words)

    ## Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #show_plot_visdom()
    #plt.show()
    #plt.close()

def evaluate_and_show_attention(input_sentence, encoder, decoder, target_sentence=None):
    output_words, attentions = evaluate(input_sentence, encoder, decoder)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
    
    #show_attention(input_sentence, output_words, attentions)
    
    ## Show input, target, output text in visdom
    #win = 'evaluted (%s)' % hostname
    #text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    #vis.text(text, win=win, opts={'title': win})
    
def evaluate_randomly(encoder, decoder):
    input_sentence, target_sentence = random.choice(pairs)
    evaluate_and_show_attention(input_sentence, encoder, decoder, target_sentence)


#def show_plot(points):
    #plt.figure()
    #fig, ax = plt.subplots()
    #loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    #ax.yaxis.set_major_locator(loc)
    #plt.plot(points)

#show_plot(plot_losses)

#output_words, attentions = evaluate("je suis trop froid .")
#plt.matshow(attentions.numpy())
#show_plot_visdom()



