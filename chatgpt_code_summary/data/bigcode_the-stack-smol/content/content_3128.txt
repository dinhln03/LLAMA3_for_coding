class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        
        result = 0
        l = len(beginWord)
        beginSet = {beginWord}
        endSet = {endWord}
        wordList = set(wordList)
        
        while beginSet or endSet:
            result += 1
            
            if len(beginSet) < len(endSet):
                beginSet, endSet = endSet, beginSet
                
            newSet = set()
                
            for word in beginSet:
                for index in range(l):
                    for c in string.ascii_lowercase:
                        newWord = word[:index] + c + word[index + 1:] 
                        
                        if newWord in endSet:
                            return result + 1
                        
                        if newWord not in wordList:
                            continue
                            
                        wordList.remove(newWord)
                        newSet.add(newWord)
                    
            beginSet = newSet
                
        return 0
        