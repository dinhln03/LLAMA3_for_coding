import sys                                                                                                      
import os                                                                                                       
import json                            
import re
import numpy as np
import pandas as pd
from Bio import motifs
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from io import StringIO

def build_mfmd_command(inputFilePath, motiflen, prb):                                                                     
    if not os.path.exists('/kb/module/work/tmp/mfmd'):                                                          
        os.mkdir('/kb/module/work/tmp/mfmd')                                                                    
    outputFilePath = '/kb/module/work/tmp/mfmd/mfmd_out/mfmd_output.txt'                                                                                                                               
    command = 'java -jar mfmd.jar ' + inputFilePath + ' ' + parameter + ' ' + prb + '  > ' + outputFilePath     
    return command                                                                                              
                                                                                                                
def run_mfmd_command(command):                                                                                 
    os.system(command)                                                                                          


def parse_mfmd_output(path):
    pfmList = []
    pfmDict={}
    outputFileList = []
    pfmMatrix=False
    seqflag=False
    motifList={}
    motifDict={}
    locList=[]
    alphabet=['A','C','G','T']

    motifSet=[]
    motifList['Condition']='temp'
    motifList['SequenceSet_ref']='123'

    background={}
    background['A']=0.0
    background['C']=0.0
    background['G']=0.0
    background['T']=0.0

    motifDict['Motif_Locations'] = []
    motifDict['PWM'] = []
    motifDict['PFM'] = []
    motiflen=0
    a=[]
    c=[]
    g=[]
    t=[]
    pwmList=[]
    pwmDict={}
    rowList = []
    rowDict={}

    for filename in os.listdir(path):
        outputFileList.append(path + '/' + filename)
        if(filename=="mfmd_out.txt"):
           outputFilePath=path+'/'+filename
           mfmdFile = open(outputFilePath,'r')
           for line in mfmdFile:
               if(re.search("PPM Matrix",line)):
                  pfmMatrix=True
               if(pfmMatrix):
                  if(line[0].isdigit()):
                     line=line.strip()
                     out=line.split()
                     pfmList.append(out)
                     a.append(out[0])
                     c.append(out[1])
                     g.append(out[2])
                     t.append(out[3])

                     rowList = []
                     rowList.append(('A',float(out[0])))
                     rowList.append(('C',float(out[1])))
                     rowList.append(('G',float(out[2])))
                     rowList.append(('T',float(out[3])))
                     rowDict['A']=float(out[0])
                     rowDict['C']=float(out[1])
                     rowDict['G']=float(out[2])
                     rowDict['T']=float(out[3])

               if(re.search("PSSM Matrix",line)):
                  pfmMatrix=False
               if(re.search("Sequences",line)):
                  seqflag=True
               if(seqflag==True):
                  line=line.strip()
                  if(re.search('\*',line)):
                     seqflag=False
                  if((line) and not (line.startswith("Seq")) and not (line.startswith("*"))):
                     line=line.rstrip()
                     seq=line.split()
                     seqid=seq[0]
                     seq_start=int(seq[1])
                     seq_end=int(seq_start)+int(motiflen)
                     sequence=seq[2]
                     orientation='+'

                     locDict={}
                     locDict['sequence_id']=seqid;
                     locDict['start']=seq_start;
                     locDict['end']=seq_end;
                     locDict['sequence']=sequence;
                     locDict['orientation']=orientation;
                     motifDict['Motif_Locations'].append(locDict)

                  
               if(re.search("Width",line)):
                  arr=line.split(" ")
                  motiflen=arr[1].split("\t")[0]

    a=[float(x) for x in a]
    c=[float(x) for x in c]
    g=[float(x) for x in g]
    t=[float(x) for x in t]
    pwmDict['A']=a
    pwmDict['C']=c
    pwmDict['G']=g
    pwmDict['T']=t

    pfmDict['A']=[]
    pfmDict['C']=[]
    pfmDict['G']=[]
    pfmDict['T']=[]

    motifStr = '>test\n'
    motifStr += 'A ' + str(a).replace(',','') + '\n'
    motifStr += 'C ' + str(c).replace(',','') + '\n'
    motifStr += 'G ' + str(g).replace(',','') + '\n'
    motifStr += 'T ' + str(t).replace(',','') + '\n'

    handle = StringIO(motifStr)

    BioMotif = motifs.read(handle, 'jaspar')
    motifDict['PWM']=pwmDict
    motifDict['PFM']=pfmDict
    motifDict['Iupac_sequence']=str(BioMotif.degenerate_consensus)
    motifSet.append(motifDict)                                            #keep in loop for multiple motifs

    motifList['Motifs']=motifSet
    motifList['Background']=background
    motifList['Alphabet']=alphabet

    return motifList

 
output=parse_mfmd_output("/home/manish/Desktop/Data/motifs/man4ish_guptamfmd/test_local/workdir/tmp/mfmd_out")

jsondata = json.dumps(output)
with open('ReportMotif.json', 'w') as outfile:
    json.dump(output, outfile)
print(jsondata)
#print(output)

