import numpy as np

import os
import time

import argparse
import PruneAndSearch as algs

def get_args():
    parser = argparse.ArgumentParser (
        prog='PruneAndSearch', 
        description='Implementation of the Prune and Search Algorithm. ',
        usage='python main.py { --rand RAND | --file FILE | --list LIST | --test [--trial TRIAL] [--vals VALS] [--verb] }  [--seed SEED]'
    )
    parser.add_argument('-n', '--small', default=None,  type=int,            help='The N-th smallest element to find in the values.    (default: {})'.format('MEDIAN'))
    parser.add_argument('-r', '--rand',  default=None,  type=int,            help='Generate N random numbers in range 1 - 10,000.      (default: {})'.format('DISABLED'))
    parser.add_argument('-f', '--file',  default=None,                       help='Read in a list from a text file.                    (default: {})'.format('DISABLED'))
    parser.add_argument('-l', '--list',  default=None,  type=int, nargs='+', help='Provide input as a list from the command line.      (default: {})'.format('DISABLED'))
    parser.add_argument('-x', '--seed',  default=123,   type=int,            help='Seed for Numpy RNG.                                 (default: {})'.format(123))
    parser.add_argument('-t', '--test',  default=False, action='store_true', help='Perform a timed test, random trials T times.        (default: {})'.format('DISABLED'))
    parser.add_argument('-T', '--trial', default=1000,  type=int,            help='Number of timed trials to conduct.                  (default: {})'.format(1000))
    parser.add_argument('-v', '--vals',  default=100,   type=int,            help='Number of random values to during testing.          (default: {})'.format(100))
    parser.add_argument('-V', '--verb',  default=False, action='store_true', help='Verbose output.                                     (default: {})'.format('DISABLED'))
    args = parser.parse_args()

    count = 0
    if args.rand != None:  count += 1
    if args.file != None:  count += 1
    if args.list != None:  count += 1
    if args.test:          count += 1

    if count >  1:         print("\n[ERROR] Too many arguments provided!!\n")
    if count == 0:         print("\n[ERROR] No arguments provided!!\n")
    if count != 1:
        parser.print_help()
        print("\n Please provide the program with an argument using one of the following:\n")
        print("\t python main.py --rand 20")
        print("\t python main.py --file a.data")
        print("\t python main.py --list 1 2 3 4 5 6 7 8")
        print("\t python main.py --test --trial 300 --vals 100 --verb --seed 123")
        print(" ")
        exit()
    
    return args



def get_list(args):

    # Simple getter function to get some list
    # based on the arguments passed in.

    if args.rand != None:
        values = np.random.randint(1, 10000, size=args.rand)
        print("Generated {} random values between 1 - 10,000.".format(args.rand))
        return values
    

    if args.file != None:
        if not os.path.exists(args.file):
            print("[ERROR] File ``{}`` does not exist!!".format(args.file))
            print("\t Please provide the path to a file.")
            exit()
        
        values = np.loadtxt(args.file, dtype=np.int32)
        return values


    if args.list != None:
        values = np.asarray(args.list, dtype=np.int32)
        return values



def test_algorithm(seed, numTrials=1000, numVals=100, maxVal=10000, verbose=True):

    # Run a series of trials on both algorithms.

    numVals = int(numVals) # 1e6
    maxVal  = int(maxVal)  # 1e10

    if verbose:
        print("\n")
        print("   --   Prune and Search Algorithm   --   ")
        print("     ================================     ")
        print("     Random Numbers Seed       =  {} ".format(seed)    )
        print("     Number of Trials          =  {} ".format(numTrials))
        print("     Number of Values in List  =  {} ".format(numVals)  )
        print("     Maximum Value in List     =  {} ".format(maxVal)   )
        print("\n")
    

    # Seed The first trial for consistency.
    np.random.seed( seed )

    # Keep a buffer of the returned finds for later comparison.
    SortAndSearchAnsBuffer  = []
    SortAndSearchTimeBuffer = []

    # Begin the trials!
    print("Beginning {} Trial on {} elements for Sort And Search   . . .  ".format(numTrials, numVals), end='', flush=True)
    for _ in range(numTrials):
        randomList = np.random.randint(maxVal, size=numVals)
        findVal    = np.random.randint(1, numVals+1)
        
        startTime  = time.time()
        ansVal     = algs.SortAndSearch(randomList, findVal)
        endTime    = time.time()
        
        SortAndSearchAnsBuffer.append(ansVal)
        SortAndSearchTimeBuffer.append( endTime - startTime )
    

    print("\u0394 : {:.4f},  \u03bc : {:.6f} \u00B1 {:.6f} ".format( 
        np.sum(  SortAndSearchTimeBuffer ), 
        np.mean( SortAndSearchTimeBuffer ), 
        np.std(  SortAndSearchTimeBuffer ) 
    ))

    

    # Seed The first trial for consistency.
    np.random.seed( seed )

    # Keep a buffer of the returned finds for later comparison.
    PruneAndSearchAnsBuffer  = []
    PruneAndSearchTimeBuffer = []

    # Begin the trials!
    print("Beginning {} Trial on {} elements for Prune And Search  . . .  ".format(numTrials, numVals), end='', flush=True)
    for _ in range(numTrials):
        randomList = np.random.randint(maxVal, size=numVals)
        findVal    = np.random.randint(1, numVals+1)

        startTime  = time.time()
        ansVal     = algs.PruneAndSearch(randomList, findVal)
        endTime    = time.time()

        PruneAndSearchAnsBuffer.append(ansVal)
        PruneAndSearchTimeBuffer.append( endTime - startTime )
    

    print("\u0394 : {:.4f},  \u03bc : {:.6f} \u00B1 {:.6f} ".format( 
        np.sum(  PruneAndSearchTimeBuffer ), 
        np.mean( PruneAndSearchTimeBuffer ), 
        np.std(  PruneAndSearchTimeBuffer ) 
    ))
    
    #for a,b in zip(SortAndSearchAnsBuffer, PruneAndSearchAnsBuffer):
    #    print(a, b, " " if a == b else "\t!!X!!")
    print("\nDid the Algorithms find the same solutions? ==> {}\n".format(PruneAndSearchAnsBuffer == SortAndSearchAnsBuffer))

    return
    


def main():

    # Fetch Arguments.
    args = get_args()

    # Seed the RNG.
    np.random.seed(args.seed)

    # Perform a timed trial and return.
    if args.test:
        test_algorithm(args.seed, numTrials=args.trial, numVals=args.vals, verbose=args.verb)
        return

    # From the args get the list.
    values = get_list(args)

    # Sent the n-value to find, median if small was not set.
    findVal = args.small if args.small != None else len(values) // 2
    

    print("\n")
    print("   --   Prune and Search Algorithm   --   ")
    print("     ================================     ")
    print("      Find The {}-Smallest Value          ".format(findVal))
    print("      In The List =                   ")
    elPerRow = 5
    for idx in range(0, len(values), elPerRow):
        print("                   ", *values[ idx : idx+elPerRow ])
    print("\n")
    
    # Naive solution in O( n log n ).
    print("Beginning Sort And Search   . . .  ", end='', flush=True)
    startTime  = time.time()
    ansVal_A   = algs.SortAndSearch(values, findVal)
    endTime    = time.time()
    print("\u0394 : {:.6f}".format( endTime - startTime ))

    print("Beginning Prune And Search  . . .  ", end='', flush=True)
    startTime  = time.time()
    ansVal_B   = algs.PruneAndSearch(values, findVal)
    endTime    = time.time()
    print("\u0394 : {:.6f}".format( endTime - startTime ))
    
    print("\nDid the Algorithms find the same solutions? ==> {}\n".format(ansVal_A == ansVal_B))
    print("The {}-Smallest Value is {}".format(findVal, ansVal_A))

    return 
       

if __name__ == '__main__':
    main()

