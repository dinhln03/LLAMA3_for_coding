
# old functions (slightly more time efficient) which store forward and reverse frames in memory and don't use
# any multiprocessing.

def seqToProtein(dnaSeq, minLen):

    newSeq = dnaSeq.upper().replace('N', '')
    start = time.time()
    forwFrames, revFrames = seqToFrames(newSeq)
    peptides = []

    for frame in forwFrames:
        peptide = tripletToAmino(frame, minLen)
        peptides += peptide

    for frame in revFrames:
        peptide = tripletToAmino(frame, minLen)
        peptides += peptide

    end = time.time()

    print(end-start)

    return peptides


def seqToFrames(dnaSeq):
    forward = dnaSeq
    reverse = createReverseSeq(dnaSeq)

    forwardFrames = createFrames(forward)
    reverseFrames = createFrames(reverse)
    return forwardFrames, reverseFrames


def createFrames(dnaSeq):
    frames = [[],[],[]]
    for i in range(0,3):
        frame = frames[i]
        for j in range(0, len(dnaSeq), 3):
            if i+j < len(dnaSeq)-2:
                triplet = dnaSeq[i+j:i+j+3]
                frame.append(triplet)


# incorporate start triplet
def tripletToAmino(frame, minLen):
    aminoList = []
    peptideList = []

    for triplet in frame:
        amino = DNA_TABLE[triplet]
        if amino == -1:
            if len(aminoList) > minLen:
                peptideList.append(''.join(aminoList))
                aminoList.clear()
        else:
            aminoList.append(amino)
    if len(aminoList) > minLen:
        peptideList.append(''.join(aminoList))
        aminoList.clear()
    return peptideList

def parseFastaDna(input_path):
    # fasta_sequences = SeqIO.parse(open(input_path), 'fasta')
    sequenceDictionary = {}
    # for fasta in fasta_sequences:
    #     name, sequence = fasta.id, str(fasta.seq)
    #     sequence = sequence.upper().replace("N", "")
    #     #sequenceDictionary[name] = sequence.upper()
    #     print(seqToProtein(sequence))
    #     break;

    with open(input_path, "rU") as handle:
        counter = 0

        # convert to tuple and look to start multiprocessing from here
        for record in SeqIO.parse(handle, 'fasta'):
            counter += 1
            print(counter)
            sequenceDictionary[record.name] = record.seq

    return sequenceDictionary

def generateOutput(outputPath, minLen, inputFile):
    finalPeptides = {}
    seqDict = parseFastaDna(inputFile)
    for key, value in seqDict.items():
        dnaSeq = str(value).upper()
        peptides = seqToProteinNew(dnaSeq, minLen)
        for peptide in peptides:
            if peptide not in finalPeptides.keys():
                finalPeptides[peptide] = [key]
            else:
                finalPeptides[peptide].append(key)
    print(finalPeptides)
    saveHandle = outputPath + '/DNAFastaProteins.fasta'
    with open(saveHandle, "w") as output_handle:
        SeqIO.write(createSeqObj(finalPeptides), output_handle, "fasta")
