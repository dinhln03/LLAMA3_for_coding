# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 17:07:18 2016

@author: sshank
"""

# Print out the required annotations at the moment... change to put into MySQL

from Bio.Seq import Seq
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
from argparse import ArgumentParser


parser = ArgumentParser()
rst_help = 'Path to parsed RST file (created with parse_rst.py).'
parser.add_argument('-r', '--rst', metavar='RST', help=rst_help, dest='rst')
input_help = 'Path to input fasta file (aligned).'
parser.add_argument('-i', '--input', metavar='INPUT', help=input_help, dest='input')
args = parser.parse_args()
rst_filename = args.rst
input_filename = args.input

descendent_sequence = ''
ancestral_sequence = ''
descendent_annotations = []
descendent_changes = []
with open(rst_filename, 'r') as file:
    for line in file:
        split = line.split()
        descendent_codon = split[6]
        ancestral_codon = split[16]
        if descendent_codon != '---':
            descendent_amino_acid = Seq(descendent_codon).translate()
            descendent_sequence += str(descendent_amino_acid)
            if descendent_codon == ancestral_codon or ancestral_codon == '---':
                # No change or missing information
                descendent_annotations.append(0)
                descendent_changes.append('-')
            else:
                ancestral_amino_acid = Seq(ancestral_codon).translate()
                if descendent_amino_acid == ancestral_amino_acid:
                    # Synonymous change
                    descendent_annotations.append(1)
                    change = ancestral_codon + '->' + descendent_codon
                    descendent_changes.append(change)
                else:
                    # Nonsynonymous change
                    descendent_annotations.append(2)
                    change = str(ancestral_amino_acid) + '->' + str(descendent_amino_acid)
                    descendent_changes.append(change)

taed_descendent = SeqRecord(descendent_sequence, id='taed_descendent')

pdb_annotations = []
pdb_changes = []
alignment = AlignIO.read(input_filename, 'fasta')
d_index = 0
p_index = 0
for k in range(alignment.get_alignment_length()):
    descendent_amino_acid, pdb_amino_acid = alignment[:, k]
    if pdb_amino_acid != '-' and descendent_amino_acid != '-':
        # There is a chance that something happened... append and increment both
        pdb_annotations.append(descendent_annotations[d_index])
        pdb_changes.append(descendent_changes[d_index])
        p_index += 1
        d_index += 1
    else:
        if pdb_amino_acid != '-':
            pdb_annotations.append(0)
            pdb_changes.append('-')
            p_index += 1
        if descendent_amino_acid != '-':
            d_index += 1

print(','.join([str(i) for i in pdb_annotations]))
print('\n')
print("'" + "','".join([str(i) for i in pdb_changes])+ "'")
