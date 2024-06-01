# problem 37
# Project Euler

__author__ = 'Libao Jin'
__date__ = 'July 17, 2015'


def rotateDigits(number):
	string = str(number)
	rotatedString = string[::-1]
	rotatedNumber = int(rotatedString)
	return rotatedNumber

def isSymmetrical(number):
	rotatedNumber = rotateDigits(number)
	if rotatedNumber == number:
		return True
	else:
		return False

def toBinary(number):
	string = bin(number)[2:]
	bNumber = int(string)
	return bNumber

def isPalindromic(number):
	if isSymmetrical(number):
		binaryNumber = toBinary(number)
		if isSymmetrical(binaryNumber):
			return True
	return False

def doubleBasePalindromes(UPPER_BOUND):
	number = 1
	DBP = []
	while number < UPPER_BOUND:
		if isPalindromic(number):
			DBP.append(number)
		number += 1
	pureDBP = DBP.copy()
	sumDBP = sum(DBP)
	for i,e in enumerate(DBP):
		DBP[i] = (e, toBinary(e))
	return (sumDBP, pureDBP, DBP)

def solution():
	UPPER_BOUND = 1000000
	DBP_Info = doubleBasePalindromes(UPPER_BOUND)
	print(DBP_Info)

solution()
