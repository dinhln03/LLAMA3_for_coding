"""
Entradas
Monto de dinero -> int -> a
"""
a = int ( input ( "Ingrese monto de dinero en COP:" ))
b = a
billetes_de_100000 = ( b - b % 100000 ) / 100000
b = b % 100000
billetes_de_50000 = ( b - b % 50000 ) / 50000
b = b % 50000
billetes_de_20000 = ( b - b % 20000 ) / 20000
b = b % 20000
billetes_de_10000 = ( b - b % 10000 ) / 10000
b = b % 10000
billetes_de_5000 = ( b - b % 5000 ) / 5000
b = b % 5000
billetes_de_2000 = ( b - b % 2000 ) / 2000
b = b % 2000
billetes_de_1000 = ( b - b % 1000 ) / 1000
b = b % 1000
monedas_de_500 = ( b - b % 500 ) / 500
b = b % 500
monedas_de_200 = ( b - b % 200 ) / 200
b = b % 200
monedas_de_100 = ( b - b % 100 ) / 100
b = b % 100
monedas_de_50 = ( b - b % 50 ) / 50
b = b % 50
print ( "La Cantidad de billetes de 100000 es de:" + str ( billetes_de_100000 ))
print ( "La Cantidad de billetes de 50000 es de:" + str ( billetes_de_50000 ))
print ( "La Cantidad de billetes de 20000 es de:" + str ( billetes_de_20000 ))
print ( "La Cantidad de billetes de 10000 es de:" + str ( billetes_de_10000 ))
print ( "La Cantidad de billetes de 5000 es de:" + str ( billetes_de_5000 ))
print ( "La Cantidad de billetes de 2000 es de:" + str ( billetes_de_2000 ))
print ( "La Cantidad de billetes de 1000 es de:" + str ( billetes_de_1000 ))
print ( "La Cantidad de monedas de 500 es de:" + str ( monedas_de_500 ))
print ( "La Cantidad de monedas de 200 es de:" + str ( monedas_de_200 ))
print ( "La Cantidad de monedas de 100 es de:" + str ( monedas_de_100 ))
print ( "La Cantidad de monedas de 50 es de:" + str ( monedas_de_50 ))