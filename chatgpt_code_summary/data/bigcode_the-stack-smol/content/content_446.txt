#!/usr/bin/python3
# --- 001 > U5W2P1_Task6_w1

def solution( n ):
    if(n > 2 and n < 7 ):
        return True;
    else:
        return False;

if __name__ == "__main__":
    print('----------start------------')
    n = 10
    print(solution( n ))
    print('------------end------------')
