#Punto 10 cambiar datos  
lista=[]
datos=(input("cantidad de datos: "))
for i in range (0,datos):
    alt=float(input("ingrese alturas: "))
    lista.append(alt)
print("la altura maxima es ", max(lista))


##################################

lista=[]
numero=int(input("numero 1 para agregar una altura y numero 2 para buscar el numero mas grande"))
n=0
while True:
    if(numero==1):
        n=float(input("altura"))
        numero=int(input("numero 1 para agregar una altura y numero 2 para buscar el numero mas grande"))
        lista.append(n)
    elif(numero==2):
        print("la mayor altura:", max(lista)) 
        break



