
from PauliInteraction import PauliInteraction
from Ising import Ising
from CoefficientGenerator import CoefficientGenerator
from Evaluator import Evaluator

#from DataVisualizer import DataVisualizer
#from DataLogger import DataLogger

import Driver as Driver_H

from MasterEquation import MasterEquation
from QuantumAnnealer import QuantumAnnealer
from Test import test
#from mpi4py import MPI


import qutip as qp
import numpy as np
import time as time
import sys

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

#NUM_TESTS = 1
#N = 7
T = 100
#NUM_INTERACTIONS = int((N * (N - 1)) / 2)
RAND_COEF = False

MASTER_RANK = 0

# MAIN TEST CASES FROM SAMPLE CODE

# THIS IS NOT USED!
def main():

    print(argv)
    COMM = MPI.COMM_WORLD
    NUM_PROC = COMM.Get_size()
    M_RANK = COMM.Get_rank()

    N = 100
    m_N = (int) (N / NUM_PROC)

    if M_RANK == MASTER_RANK:
        A = np.arange(N, dtype = np.float64)
        start_time = MPI.Wtime()
    else:
        A = np.empty(N, dtype = np.float64)

    m_A = np.empty(m_N, dtype = np.float64)

    # Scatter
    COMM.Scatter([A, MPI.DOUBLE], [m_A, MPI.DOUBLE])

    for i in range(m_N):
        m_A[i] = M_RANK

    COMM.Barrier()

    COMM.Allgather([m_A, MPI.DOUBLE], [A, MPI.DOUBLE])

    COMM.Barrier()

    if M_RANK == MASTER_RANK:
        print(A)

    #for i in range(10):
        #test(RAND_COEF, i)

# THIS IS USED!
def parallel_main(NUM_TESTS, NUM_QUBITS):
    
	# MPI Initialization
    COMM = MPI.COMM_WORLD
    NUM_PROC = COMM.Get_size()
    M_RANK = COMM.Get_rank()

	# If process is master rank, allocate array of overlap probabilities
    if M_RANK == MASTER_RANK:
        overlap_probabilities = np.zeros(NUM_TESTS, dtype = np.float64)
        start_time = MPI.Wtime()
    else:
        overlap_probabilities = np.empty(NUM_TESTS, dtype = np.float64);

	# Calculate the local number of tests to perform
    M_TESTS = (int) (NUM_TESTS / NUM_PROC)
	
	# Allocate local overlap probablity arrays
    m_overlap_probabilities = np.empty(M_TESTS, dtype = np.float64)

	# Scatter the global overlap probabilities
    COMM.Scatter([overlap_probabilities, MPI.DOUBLE], [m_overlap_probabilities, MPI.DOUBLE])

	# And for each process, perform its local tests and save overlap probability
    for i in range(M_TESTS):
        m_overlap_probabilities[i] = test(RAND_COEF, 0, NUM_QUBITS)

	# Enforce synchronization
    COMM.Barrier()

	# Gather the local overlap probabilities in master rank
    COMM.Allgather([m_overlap_probabilities, MPI.DOUBLE], [overlap_probabilities, MPI.DOUBLE])

	# Enforce synchronization
    COMM.Barrier()

	# When tests are done, master rank will process data and print
    if M_RANK == MASTER_RANK:
        stop_time = MPI.Wtime()
        total_time = stop_time - start_time
		# Print probabilities - TODO(Log this to a file, not just print to screen)
        #for i in range(len(overlap_probabilities)):
            #print("ITERATION %d, OVERLAP PROBABILITY = %f" % (i, overlap_probabilities[i]))
        
		# Print run statistics
        print("---------- NUMBER OF QUBITS = %d ----------" % NUM_QUBITS)
        print("\tNUMBER OF PROCESSES = %d" % NUM_PROC)
        print("\tNUMBER OF TESTS = %d" % NUM_TESTS)
        print("\tTOTAL TIME = %f sec" % total_time)
        print("------------------------------------------")

# Initial script
if __name__ == "__main__":
    NUM_TESTS = int(sys.argv[1])
    NUM_QUBITS = int(sys.argv[2])
    parallel_main(NUM_TESTS, NUM_QUBITS)


