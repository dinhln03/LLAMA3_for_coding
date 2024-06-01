def palindrome(str):
    end = len(str)
    middle = end >> 1
    for i in range(middle):
        end -= 1
        if(str[i] != str[end]):
            return False
    return True
while True:
    word = input('Enter word: ')
    if word == 'done' : break
    palindrome(word)
    if palindrome(word) == True:
        print('Palindrome')
    else:
        print('No Palindrome')
