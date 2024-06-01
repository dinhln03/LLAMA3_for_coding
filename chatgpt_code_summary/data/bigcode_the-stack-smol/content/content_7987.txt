# INIT
data = []
numeric = []
normal = []
keyL = []

KeyL = []

key = input("Enter Key Value: ")

# File - Load Function
def load(file):
    handle = open(file)
    return handle.read()

# Text Format
def form(file):
    format = load(file)
    format = format.replace(' ', '')
    format = format.replace(',', '')
    format = format.replace('-', '')
    format = format.replace('–', '')
    format = format.replace('—', '')
    format = format.replace('.', '')
    format = format.replace(';', '')
    format = format.replace('\n', '')
    return format

    #ADDS TO LIST
    for letter in format:
        data.append(letter)

    #REMOVE NUM
    for letter in data:
        global numbers
        numbers = ""
        
        if not letter.isdigit():
            normal.append(letter)
        else:
            numeric.append(letter)
            numbers = ''.join(numeric)
            
    return format, numbers

#Mod Inv
def modInverse(a, m):
    a = a % m
    for x in range(1, m):
        if ((a * x) % m == 1):
            return x
    return 1

#Calc dif
def dif(a,b):
    if a > b:
        return a - b
    if a < b:
        return b - a
    else:
        return 0


#Key Creator
def getKey(key):
    lenKey = len(key)
    lenPtext = len(normal)
    difP = lenPtext/lenKey  #Calc diff of Plain text 
    if difP % 1 == 0:
        KEY = ""
        difP = difP
        KEY = key*difP
        keyL.append(KEY)
    else:
        KEY = ""
        difP = int(difP)+1
        KEY = key*difP
        keyL.append(KEY)

    for word in keyL:
        for letter in word:
            KeyL.append(letter)

    i = 0
    for i in range(2):
        print(i)
    print("test")
   
form('project2plaintext.txt.txt')
#print(len(normal))
#print(numbers)
#print(dif(len(getKey(key)),len(normal)))
getKey(key)
print(KeyL)
