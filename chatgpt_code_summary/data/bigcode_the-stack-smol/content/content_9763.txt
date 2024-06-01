def find_words(string, word_set):
    if string == "" or not word_set:
        return None
    if string in word_set: # O(1)
        return [string]
    #"bedbathbeyondunk"
    #{'bed', 'bath', 'bedbath', 'and', 'beyond'}

    tmp = "" # bedbathbeyondunk
    out = [] # []
    retro = False # True
    i = 0
    while i < len(string): # i = 15
        if not retro:
            tmp += string[i]

        if tmp in word_set:
            out.append(tmp)
            tmp = ""

        if i == len(string)-1 and tmp != "":
            if not out:
                return None
            tmp = out.pop() + tmp
            retro = True
            i -= 1
        i += 1

    return out 


assert find_words(
    "bedbathandbeyond", 
    set(['bed', 'bath', 'bedbath', 'and', 'beyond'])
) == ['bed', 'bath', 'and', 'beyond']

assert find_words(
    "thequickbrownfox", 
    set(['quick', 'brown', 'the', 'fox'])
) == ['the', 'quick', 'brown', 'fox']

assert find_words(
    "thequickbrownfoxa", 
    set(['quick', 'brown', 'the', 'fox'])
) == None
