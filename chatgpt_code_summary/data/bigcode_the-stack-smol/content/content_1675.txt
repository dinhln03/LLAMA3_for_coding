import string
import sbxor

"""
Detect single-character XOR

One of the 60-character strings in this file (4.txt) has been encrypted by single-character XOR.

Find it.
"""

if __name__ == "__main__":
    with open("data/4.txt", "r") as data_file:
        data = data_file.read().split("\n")

    candidates = []
    for line in data[:]:
        line_byte = bytearray.fromhex(line)
        sb = sbxor.solve(line_byte)
        if len(sb) != 0:
            candidates.append([line_byte, sb])
    
    print(f"{len(candidates)} candidate(s) found for single-byte xor\n")
    for candidate in candidates:
        print(f"Ciphertext: {candidate[0]}")
        print("Possible solution(s):")
        for b in candidate[1]:
            print(f"Key: {b[0]}")
            print(f"Plaintext: {repr(b[1])}")
