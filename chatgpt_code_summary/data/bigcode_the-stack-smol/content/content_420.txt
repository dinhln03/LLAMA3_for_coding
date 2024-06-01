"""
This file implements the signature scheme from "Unique Ring Signatures: A Practical
Construction" by Matthew Franklin and Haibin Zhang
"""

import sys
import math
from random import randint
import hashlib
from libsig.AbstractRingSignatureScheme import AbstractRingSignatureScheme
#from AbstractRingSignatureScheme import AbstractRingSignatureScheme
#from libsig import primes


# ----------- HELPER FUNCTIONS ----------- 

# function to find divisors in order to find generators
def find_divisors(x):
    """
    This is the "function to find divisors in order to find generators" module.
    This DocTest verifies that the module is correctly calculating all divisors
    of a number x.

    >>> find_divisors(10)
    [1, 2, 5, 10]

    >>> find_divisors(112)
    [1, 2, 4, 7, 8, 14, 16, 28, 56, 112]
    """
    divisors = [ i for i in range(1,x+1) if x % i == 0]
    return divisors

# function to find random generator of G
def find_generator(p):
    '''
    The order of any element in a group can be divided by p-1.
    Step 1: Calculate all Divisors.
    Step 2: Test for a random element e of G wether e to the power of a Divisor is 1.
            if neither is one but e to the power of p-1, a generator is found.
    '''

    # Init
    # Generate element which is tested for generator characteristics.
    # Saved in list to prevent checking the same element twice.
    testGen = randint(1,p)
    listTested = []
    listTested.append(testGen)

    # Step 1.
    divisors = find_divisors(p)

    # try for all random numbers
    # Caution: this leads to a truly random generator but is not very efficient.
    while len(listTested) < p-1:
        # only test each possible generator once
        if testGen in listTested:
            # Step 2.
            for div in divisors:
                testPotency = math.pow(testGen,div) % (p+1)
                if testPotency == 1.0 and div != divisors[-1]:
                    # element does not have the same order like the group,
                    # therefore try next element
                    break
                elif testPotency == 1.0 and div == divisors[-1]:
                    # generator is found
                    return testGen
        # try new element
        testGen = randint(1,p)
        listTested.append(testGen)

def list_to_string(input_list):
    '''
    convert a list into a concatenated string of all its elements
    '''

    result = ''.join(map(str,input_list))
    return result


# ----------- HELPER FUNCTIONS END ----------- 


class UniqueRingSignature(AbstractRingSignatureScheme):
    '''
    | output: pp = (lamdba, q, G, H, H2) with,
    | q is prime,
    | g is generator of G,
    | G is multiplicative Group with prime order q,
    | H1 and H2 are two Hash functions H1: {0,1}* -> G,
    | (as well as H2: {0,1}* -> Zq which is the same).
    '''

    # set prime p (Sophie-Germain and therefore save)
    #q = 53
    q = 59
    # find random generator of G
    g = find_generator(q-1)

    # hash functions with desired range and the usage of secure hashes
    h1 = lambda x: int(hashlib.sha256(str(x).encode()).hexdigest(),16)%(UniqueRingSignature.q)
    # this way to share the information should be improved
    h2 = lambda x: int(hashlib.sha512(str(x).encode()).hexdigest(),16)%(UniqueRingSignature.q)

    # list of public keys
    Rp = list()

    @staticmethod
    def keygen(verbose=False):
        #print("---- KeyGen Started  ---- \n")
        r = randint(1,UniqueRingSignature.q)
        # x = g**r % q
        x = pow(UniqueRingSignature.g, r,UniqueRingSignature.q)

        # y = g**x
        y = pow(UniqueRingSignature.g, x, UniqueRingSignature.q)

        if verbose == True:
            print("KeyGen Config: public key y=" + str(y) + ", private key x=" + str(x) + "\n")
            print("---- KeyGen Completed ---- \n")
        # Caution! I know, keygen should NOT return the private key, but this is needed to "play" through a whole signature - validation process
        return x,y

    @staticmethod
    def ringsign(x, pubkey, message,verbose=False):
        '''
        input: x is the privkey from user i, 
              | all public keys: pubkeys,
              | the message
         
        output: (R,m, (H(mR)^xi), c1,t1,...,cn,tn),
              | R: all the pubkeys concatenated,
              | cj,tj: random number within Zq
        ''' 

        # calculate R = pk1,pk2,..,pkn
        R = list_to_string(pubkey)

        g = UniqueRingSignature.g
        q = UniqueRingSignature.q
        h1 = UniqueRingSignature.h1
        h2 = UniqueRingSignature.h2

        # message + pubkeys concatenated
        mR = message + str(R)

        C = list()
        T = list()
        A = list()
        B = list()
        ri = -1

        # simulation step
        #
        for i in pubkey:
            # Step 1:
            # 
            a = 0 
            b = 0
            c = 0
            t = 0
            if pow(g,x,q) != i:
                c, t = randint(1,q), randint(1,q)
                a = (pow(g, t) * pow(int(i), c)) % q
                b = (pow(h1(mR), t) * pow(pow(h1(mR),x),c)) % q
            else:
                # Step 2:
                # 
                ri = randint(1, q)
                a = pow(g, ri, q)
                b = pow(h1(mR), ri, q)

                # insert to allocate place
                c = -1
                t = -1

            A.append(a)
            B.append(b)
            C.append(c)
            T.append(t)
        # for end

        # Step 3:
        # 
        cj = 0

        # list count from 0
        ab = ''.join('{}{}'.format(*t) for t in zip(A,B))

        usernr = 0
        for i in range(len(pubkey)):
            if pubkey[i] != (pow(g,x,q)):
                cj = (cj + C[i]) % q
            else: 
                usernr = i

        ci = h2(message + R + ab) - (cj % (q-1))

        # update ci, this was initialized with -1
        C[usernr] = ci

        ti = ((ri - (C[usernr]*x)) % (q-1))
        if ti < 0:
            ti = (q-1) + ti

        # update ti, this was initialized with -1
        T[usernr] = ti

        # Step 4:
        # 
        # concatenate ct: c1,t1,c2,t2,...,cn,tn
        ct = ','.join('{},{}'.format(*t) for t in zip(C,T))

        # returning result
        result = R + ","+message+","+str(pow(h1(mR),x, q))+"," + ct 
        if verbose == True:
            print("RingSign Result: "+ result)
            print("---- RingSign Completed ---- \n")
        return result


    @staticmethod
    def verify(R, message, signature,verbose=False):
        '''
        Input: the public keys R
        |       the message
        |       the signature computed with ringsign

        Output: whether the message was signed by R or not
        '''

        g = UniqueRingSignature.g
        q = UniqueRingSignature.q
        h1 = UniqueRingSignature.h1
        h2 = UniqueRingSignature.h2

        # parse the signature
        parsed = signature.split(",")
        tt = int(parsed[2])
        cjs = list()
        tjs = list()
        for i in range(0,int(((len(parsed))/2)-1)):
            cjs.append(int(parsed[3+2*i]))
            tjs.append(int(parsed[4+2*i]))

        #print(str(cjs)+"  "+str(tjs) + "   "+ str(tt))

        # check signature
        # sum of all cjs
        # =?
        # self.pp['h2'](message + R + gyh1)

        mR = list_to_string(R)

        val1 = sum(cjs) % q
        # for all users in R:
        # g**tj * yj ** cj , h1(m||R)**tj * tt**cj
        gyh1 = ""
        for i in range(len(tjs)):
            if tjs[i] < 0:
                tjs[i] = (q-1) + tjs[i]
            if cjs[i] < 0:
                cjs[i] = (q-1) + cjs[i]

            gy = (pow(g,(tjs[i]),q) * (pow((R[i]),(cjs[i]),q))) % q
            h = (pow(int(h1(message + mR)), int(tjs[i])) * pow(tt,int(cjs[i]))) % q
            gyh1 = gyh1 + str( gy) + str( h)

        val2 = str(h2(message + list_to_string(R) + gyh1))
        if int(val1) == int(val2):
            if verbose == True:
                print("Signature is valid!\n")
                print("Common Result: " + str(val1))
                print("---- Validation Completed ---- \n")
            return True
        else:
            if verbose == True:
                print("Signature is not valid!\n")
                print(str(val1) + " != " + str(val2))
                print("---- Validation Completed ---- \n")
            return False                                                              

def local_test(verbose=True):
    # verbose output
    print(verbose)
    
    # user 1 will signate and validate later,
    # therefore his private key is saved for test purposes
    privKey1,pubkey = UniqueRingSignature.keygen(verbose)
    UniqueRingSignature.Rp.append(pubkey)
    a,pubkey = UniqueRingSignature.keygen(verbose)
    UniqueRingSignature.Rp.append(pubkey)


    # usernr start from 0
    # ringsign(self, privkey, usernr, pubkeys, message)
    ring = UniqueRingSignature.ringsign(privKey1, UniqueRingSignature.Rp, "asdf", verbose)
    if verbose:
        print("Result of Signature Validation:")
    # verify(pubkeys, message, signature):
    UniqueRingSignature.verify(UniqueRingSignature.Rp, "asdf", ring, verbose)

if __name__ == '__main__':
    # doctest start
    import doctest
    doctest.testmod()

    if len(sys.argv) > 1:
        verbose = False
        if sys.argv[1] == "True":
            verbose = True
        # run a local test
        local_test(verbose)
    
