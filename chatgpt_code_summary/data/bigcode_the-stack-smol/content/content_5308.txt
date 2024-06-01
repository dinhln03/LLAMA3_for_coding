""" main_MST.py

Skript illustrating the calculation of a minimum weight spanning tree for an
exemplary graph instance.

Supplemental Material for the Lecture Notes "Networks - A brief Introduction
using a Paradigmatic Combinatorial Optimization Problem" at the international
summer school "Modern Computational Science 10 - Energy of the Future" held
in Oldenburg, September 3-14, 2018

Author: O. Melchert
Date: 2018-09-11
"""
import sys
from LibMCS2018 import *

def main():
    G = fetchWeightedGraph(sys.argv[1])
    T,Twgt = mstKruskal(G)
    print mstGraphviz(G,T)

main()
# EOF: main_MST.py
