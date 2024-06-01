import hashlib
import requests

import sys


def valid_proof(last_proof, proof):
    guess = f'{last_proof}{proof}'.encode()
    guess_hash = hashlib.sha256(guess).hexdigest()
    
    return guess_hash[:6] == "000000"

def proof_of_work(last_proof):
    """
    Simple Proof of Work Algorithm
    - Find a number p' such that hash(pp') contains 6 leading
    zeroes, where p is the previous p'
    - p is the previous proof, and p' is the new proof
    """
    print(f'\nSearch for proof initialized.\n')

    proof = 0
    while valid_proof(last_proof, proof) is False:
        proof += 1

    print(f'\nSearch for proof complete, proof is {proof}\n')
    
    return proof 


if __name__ == '__main__':
    # What node are we interacting with?
    if len(sys.argv) > 1:
        node = sys.argv[1]
    else:
        node = "http://localhost:5000"

    coins_mined = 0
    # Run forever until interrupted
    while True:
        # Get the last proof from the server and look for a new one
        proof = requests.get(url=node + '/last_proof')
        new_proof = proof_of_work(proof.json()['proof'])

        # When found, POST it to the server {"proof": new_proof}
        data = {'proof': new_proof}
        attempt = requests.post(url=node + '/mine',json=data)
        # If the server responds with 'New Block Forged'
        if attempt.json()['message'] == 'New Block Forged':
            # add 1 to the number of coins mined and print it.
            coins_mined += 1
            print("TOTAL COINS MINED:", coins_mined)
        else:
            # else print the message from the server.
            print(attempt.json()['message'])
