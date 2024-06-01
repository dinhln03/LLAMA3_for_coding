import operator

class Istr:

    def count(self, s, k):
        letters = {}
    
        for letter in s:

            if letter not in letters:
                letters[letter] = 1 
            else:
                letters[letter] += 1
        
        for i in range(0, k):
            
            index =  max(letters.iteritems(), key=operator.itemgetter(1))[0] 
            letters[index] -= 1 
        
        score = 0
        for element in letters:
            val = letters[element] * letters[element]

            score += val
        return score
