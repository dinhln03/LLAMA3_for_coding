import re
from Error import Error4,Error6,Error9
from DataBase import BaseDatos,BdRow
from .precompilada import precompilada
from typing import Pattern

def getEtiqueta(linea:str)->str:
    """Obtiene el nombre de la captura

    Args:
        linea (str): Linea donde se va a buscar la etiqueta

    Returns:
        str: Regresa el nombre de la etiqueta
    """

    # Buscamos el mnemonico
    pattern='\s+([a-z]{1,5})\s+([a-z]{1,24})'     
    busqueda=re.search(pattern,linea,re.IGNORECASE)

    # Obtenemos el mnemonico-------------------------------
    etiqueta =busqueda.group(2)
    return etiqueta


def calcularEtiqueta(sustraendo:str,minuendo:str)-> str:
    """Resta la diferencia entre dos PC en hexadecimal
    sustraendo - minuendo

    - Si
    - Sustraendo - minuendo
    - En caso de error regresa 'e10' operando muy grande

    Args:
        sustraendo (str): Ejemplo '0x7'
        minuendo (str): Ejemplo '0x1'

    Returns:
        str: Ejemplo  '0x06'
    """
    print(sustraendo)
    print(minuendo)
    sustraendo=int(sustraendo,16)
    minuendo=int(minuendo,16)

    resultado:int= sustraendo-minuendo
    print(resultado)
    if resultado <-127 or 128<resultado:
        return 'e10' #E10 el salto relativo es muy lejano  
        # Si es negativa
    elif resultado<0:
        return convertirA2Hex(resultado)
        # si es positiva
    else:
        return hex(resultado)


def bindigits(n:int, bits:int)->str:
    """Convierte a binario un numero de complemento A2 en caso de negativo, normal en caso de ser positivo

    Args:
        n (int): E.g 7
        bits (int): eg 3

    Returns:
        str: E.g '001'
    """
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)


def convertirA2Hex(numero:int)-> str:
    """Convierte un numero decimal a hexadecimal

    - Si el número es decimal lo convierte a complemento A2

    Args:
        numero (int): Número decimal que se quiere convertir Eg. 07

    Returns:
        str: Eg. 0x07
    """
    # cuantos bits ocupa el número hexadecimal
    cuantosBits=(len(hex(numero))-2) *4 # el -2 es 0x, el 4 es porque 1 hex equivale a 4 bits

    #numero convertido a binario
    binario=bindigits(numero,cuantosBits)

    return hex(int(binario, 2))



def precompilarPasada1(numLinea:int,modo:str,linea:str,pc: str)->precompilada:
    # variables globales
    
    # Buscamos el mnemonico
    pattern='\s+([a-z]{1,5})\s+([a-z]{1,24})'     
    busqueda=re.search(pattern,linea,re.IGNORECASE)

    # Obtenemos el mnemonico-------------------------------
    mnemonico =busqueda.group(1)
    etiqueta=busqueda.group(2)

    # Consulta a la base de datos-------------------------------
    consultaBd:BdRow = BaseDatos.bdSearch(mnemonico,6)

    # obtenemos el Pc Actual=pc + bytesOcupados
    pcActual=hex(int(pc,16) +2) # El más 2  es porque todas las relativos usan 2 bytes

    # Datos directos--------------------------------------
    lineaPrecompilada=precompilada(numLinea,modo,pcActual,consultaBd.opcode,etiqueta,consultaBd.byte)

    # Datos detivados-----------------------------------
    lineaPrecompilada.bytesOcupados=consultaBd.byte

    return lineaPrecompilada

def precompilarPasada2(lineaPrecompilada:precompilada,pcEtiqueta:str)->precompilada:
    # obtenemos el Pc Actual=pc + bytesOcupados
    pcActual=hex(int(lineaPrecompilada.pcActual,16) ) # El más 2  es porque todas las relativos usan 2 bytes

    lineaPrecompilada1:precompilada

    # Calculamos el operando
    operandoPrecompilado=calcularEtiqueta(pcEtiqueta,pcActual)

    # Verificamos si el salto relaitvo no es tan grande
    if operandoPrecompilado=='e10': # en caso de error salto muy lejando
        lineaPrecompilada1=precompilada(0,'','','','',0)
        lineaPrecompilada1.error='e10'
    else:
        operandoPrecompilado=operandoPrecompilado[2:]


        # hacer una copia
        lineaPrecompilada1=precompilada(lineaPrecompilada.numLinea,lineaPrecompilada.modo,hex(int(lineaPrecompilada.pcActual,16)-2),lineaPrecompilada.opcode,operandoPrecompilado,lineaPrecompilada.byte)

    print(operandoPrecompilado)

    return lineaPrecompilada1
    #return lineaPrecompilada1