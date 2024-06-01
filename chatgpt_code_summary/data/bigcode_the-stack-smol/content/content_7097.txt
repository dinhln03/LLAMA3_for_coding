#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:37:19 2018

@author: kaustabh
"""




#making a single list representing all sides in a serial fashion

r = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','b1','b2','b3','r1','r2','r3','g1','g2','g3','o1','o2','o3','b4','b5','b6','r4','r5','r6','g4','g5','g6','o4','o5','o6','b7','b8','b9','r7','r8','r9','g7','g8','g9','o7','o8','o9', 'y1','y2','y3','y4','y5','y6','y7','y8','y9']

real = r[:] 


# cube is the list and we can't use random.shuffle because certain colors stay together
    


def shift(r,L): #r is cube list ; L is 12 element array to be shifted
      
    t1 = r[L[0]]
    t2 = r[L[1]]
    t3 = r[L[2]]
    
    r[L[0]] , r[L[3]] = r[L[3]] , r[L[0]]
    r[L[1]] , r[L[4]] = r[L[4]] , r[L[1]]
    r[L[2]] , r[L[5]] = r[L[5]] , r[L[2]]
     
    r[L[3]] , r[L[6]] = r[L[6]] , r[L[3]]
    r[L[4]] , r[L[7]] = r[L[7]] , r[L[4]]
    r[L[5]] , r[L[8]] = r[L[8]] , r[L[5]]
    
    r[L[6]] , r[L[9]] = r[L[9]] , r[L[6]]
    r[L[7]] , r[L[10]] = r[L[10]] , r[L[7]]
    r[L[8]] , r[L[11]] = r[L[11]] , r[L[8]]
       
    r[L[9]] = t1
    r[L[10]] = t2
    r[L[11]] = t3
    
    return r

def rotate(r, L): #a face will also rotate
    
    t1 = r[L[0]]
    t2 = r[L[7]]
    
    r[L[0]] , r[L[6]] = r[L[6]] , r[L[0]]
    r[L[7]] , r[L[5]] = r[L[5]] , r[L[7]]
    r[L[6]] , r[L[4]] = r[L[4]] , r[L[6]]
    r[L[3]] , r[L[5]] = r[L[5]] , r[L[3]]
    r[L[2]] , r[L[4]] = r[L[4]] , r[L[2]]
    r[L[1]] , r[L[3]] = r[L[3]] , r[L[1]]
    r[L[2]] = t1
    r[L[1]] = t2
    
    return 
    
    

def right_c(r):
    
    L = [33,34,35,36,37,38,39,40,41,42,43,44]
    rt = [45,48,51,52,53,50,47,46]
    rotate(r,rt)
    return shift(r, L)

def right_ac(r):
    
    L = [33,34,35,36,37,38,39,40,41,42,43,44]
    L.reverse()
    rt = [45,48,51,52,53,50,47,46]
    rt.reverse()
    rotate(r,rt)
    return shift(r, L)

def left_ac(r):
    
    rt = [0,1,2,5,8,7,6,3]
    rotate(r,rt)
    L = [9,10,11,12,13,14,15,16,17,18,19,20]
    return shift(r, L)

def left_c(r):
    
    rt = [0,1,2,5,8,7,6,3]
    rt.reverse()
    rotate(r,rt)
    L = [9,10,11,12,13,14,15,16,17,18,19,20]
    L.reverse()
    return shift(r, L)

def up_c(r):
    
    rt = [9,21,33,34,35,23,11,10]
    rotate(r, rt)
    L = [0,3,6,12,24,36,45,48,51,18,30,42]
    return shift(r, L)

def up_ac(r):
    
    rt = [9,21,33,34,35,23,11,10]
    rt.reverse()
    rotate(r, rt)
    L = [0,3,6,12,24,36,45,48,51,18,30,42]
    L.reverse()
    return shift(r, L)

def down_ac(r):
    
    rt = [15,16,17,29,41,40,39,27]
    rotate(r,rt)
    L = [2,5,8,14,26,38,47,50,53,20,32,44]
    return shift(r, L)

def down_c(r):
    
    rt = [15,16,17,29,41,40,39,27]
    rt.reverse()
    rotate(r,rt)
    L = [2,5,8,14,26,38,47,50,53,20,32,44]
    L.reverse()
    return shift(r, L)

def front_c(r):
    
    rt = [12,24,36,37,38,26,14,13]
    rotate(r,rt)
    L = [11,23,35,45,46,47,39,27,15,8,7,6]
    return shift(r, L)

def front_ac(r):

    rt = [12,24,36,37,38,26,14,13]
    rt.reverse()
    rotate(r,rt)
    L = [11,23,35,45,46,47,39,27,15,8,7,6]
    L.reverse()
    return shift(r, L)

def back_c(r):
    
    rt= [18,30,42,43,44,32,20,19]
    rotate(r,rt)
    L = [0,3,6,9,21,33,45,48,51,15,27,39]
    return shift(r, L)

def back_ac(r):
    
    rt= [18,30,42,43,44,32,20,19]
    rt.reverse()
    rotate(r,rt)
    L = [0,3,6,9,21,33,45,48,51,15,27,39]
    L.reverse()
    return shift(r, L)

def shuffle(r):
    
    import random
    for i in range(random.randint(17,32)):
        random.choice([right_c(r),left_c(r),up_c(r),down_c(r),front_c(r),back_c(r),back_ac(r),right_ac(r),left_ac(r),up_ac(r),front_ac(r),down_ac(r)])
    return r

def reset():
    
    return real[:]

def display(r):
    for j in range(15):
          for i in range(9):
              if j == 0 :
                  if i == 0 :
                     print(" ")
                  while i<4:
                       print(" ", end='')
                       i += 1
                  if i == 5:
                     print(r[9]+" "+r[21]+" "+r[33],end='\n')
                     
              if j == 1 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[10]+" "+r[22]+" "+r[34],end='\n')
                     

              if j == 2 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[11]+" "+r[23]+" "+r[35],end='\n')
                     
              if j == 3 :
                  if i == 0:
                   print(" ")
                     
              if j == 4 :
                  if i==0 :
                     print(r[0]+" "+r[3]+" "+r[6]+"  ",end='')
                  if i==3 :
                     print(r[12]+" "+r[24]+" "+r[36]+" ",end='')
                  if i==6 :
                      print(" "+r[45]+" "+r[48]+" "+r[51],end='\n')
                      
              if j == 5 :
                  if i==0 :
                     print(r[1]+" "+r[4]+" "+r[7]+"  ",end='')
                  if i==3 :
                     print(r[13]+" "+r[25]+" "+r[37]+" ",end='')
                  if i==6 :
                      print(" "+r[46]+" "+r[49]+" "+r[52],end='\n')
            
              if j == 6 :
                  if i==0 :
                     print(r[2]+" "+r[5]+" "+r[8]+"  ",end='')
                  if i==3 :
                     print(r[15]+" "+r[26]+" "+r[38]+" ",end='')
                  if i==6 :
                      print(" "+r[47]+" "+r[50]+" "+r[53],end='\n')
                      
              if j == 7 :
                  if i==0:
                   print(" ")
                      
              if j == 8 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[15]+" "+r[27]+" "+r[39],end='\n')
                
              if j == 9 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[16]+" "+r[28]+" "+r[40],end='\n')
                     
              if j == 10 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[17]+" "+r[29]+" "+r[41],end='\n')
                     
              if j == 11 :
                  if i==0 :
                   print(" ")
            
              if j == 12 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[18]+" "+r[30]+" "+r[42],end='\n')
                     
              if j == 13 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[19]+" "+r[31]+" "+r[43],end='\n')
                     
              if j == 14 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[20]+" "+r[32]+" "+r[44],end='\n')
                  if i == 8:
                      print(" ")
                      print(" ------------------------------------- ")
            
    return ' '.join(r)

print(r)
display(r)
left_ac(r)
left_ac(r)
display(r)
left_ac(r)
left_ac(r)
display(r)



    






    
    



