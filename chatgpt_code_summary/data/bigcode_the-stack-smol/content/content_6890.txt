import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
f_out = "E:\\1\\P_rk4.txt" # address file for output
f2 = open(f_out,"w+")
def du_dx(x,y):
    wa=1      # atomic frequency  
    wp=0.6    # field frequency
    g=0.6    # coupling strength 
    n = 1     # number of photons
    A = n*wp+(wa/2)
    B = (1+n)*wp-(wa/2)
    X = n+1
    C = np.sqrt(X)    
    dydx_1=  A*y[1]+g*C*y[3]
    dydx_2= -A*y[0]-g*C*y[2]
    dydx_3=  B*y[3]+g*C*y[1]
    dydx_4= -B*y[2]-g*C*y[0]  
    return [dydx_1,dydx_2,dydx_3,dydx_4]

y_0 = (1/np.sqrt(2),0,1/np.sqrt(2),0) # initial value
# print("y_0 = ",y_0)
m = 1000
ti = 0
tf = 30
h = tf/m
tspan = np.arange(ti,tf,h)
print(h)
for i in tspan:
    print(i)
    v = RK45(du_dx,t0 =i,y0 = y_0,t_bound=i) # 4 answer of dydx_1,...,dydx_4
    print(v.y[0:])
# print(type(v))

# print("v.t[0] = ",v.t[0])
# print(len(v.t))
# print("------------------")
# print(v.y)
# print(len(v.t))
# print("------------------")
# y_1 = v.y[:,0]
# print("y_1 = ",y_1)
# print("------------------")
# y_2 = v.y[0,:]
# print("y_2 = ",y_2)
# print("------------------")
# y_3 = v.y[0,0]
# print("y_3 = ",y_3)
# print("------------------")
# # --------------------------
# # print in file 
# count = 0
# while count<1000:
#     y_i = v.y[:,count]
#     f2.write(str(v.t[count]))
#     f2.write("     ")
#     for i in y_i:
#         i = round(i,4)
#         i = str(i)
#         f2.write(i)
#         f2.write(len(i)*" ")
#     f2.write("\n")
#     count = count+1

# # y_prime = u_s[:,1]
# # print(y_prime)
# plt.plot(v.t, v.y[0,:],'-', label='r(t)') 
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()