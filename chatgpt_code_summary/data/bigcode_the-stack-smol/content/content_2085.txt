def appendAndDelete(s, t, k):
    iter=0
    s=[]
    t=[]
    
    
    
    while s:
        s.pop(0)
        iter+=1
    for i in t:
        s.append(i)
        iter+=1
    if iter==k:
        print("Yes")
    else:
        print("No")
        
