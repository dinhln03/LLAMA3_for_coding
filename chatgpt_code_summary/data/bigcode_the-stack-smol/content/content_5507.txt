for tt in range(int(input())):
    n = int(input())
    ans = 0 
    arr = list(map(int,input().split()))
    freq = dict()

    for i in range(n):
        arr[i] -= i  
        if arr[i] not in freq:
            freq[arr[i]] = 0 
        freq[arr[i]] += 1 
    for i in freq:
        ans += freq[i]*(freq[i]-1)//2  
        
    print(ans)