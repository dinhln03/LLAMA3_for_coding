class Solution(object):
    def changebase(self, n, base):
    	digits = "0123456789ABCDEF"
    	remstack = []
    	while n > 0:
    		rem = n % base 
    		remstack.append(rem)
    		n = n / base

    	newString = ""
    	while not len(remstack) == 0:
    		newString += digits[remstack.pop()]
    	return newString

    def countNum(self, n, base):
    	res = self.changebase(n, base)[:-1][::-1]
    	i = 0
    	count = 0
    	while i < len(res):
    		print int(res[i])
    		count += base**i * int(res[i])
    		i += 1
    	print count 

    	

a = Solution()
print a.changebase(44, 4)
print a.countNum(44, 4)
print a.changebase(23, 4)