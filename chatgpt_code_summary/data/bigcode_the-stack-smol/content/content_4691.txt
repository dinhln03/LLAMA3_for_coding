#python exceptions let you deal with
#unexpected results

try:
	print(a)  #this will throw an exception since a is not found
except:
	print("a is not defined!") 

#there are specific errors in python
try:
	print(a) #this will throw a NameError
except NameError:
	print("a is still not defined") 
except:
	print("Something else went wrong.")

#this will break our program
#since a is not defined
print(a)