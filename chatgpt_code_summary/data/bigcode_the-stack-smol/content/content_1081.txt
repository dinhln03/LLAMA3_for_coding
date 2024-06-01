'''

TRIES

Trie support search, insert, and deletion in O(L) time where L is length of the key

why Trie?
    * With Trie, we can insert and find strings in O(L) time where L represent the length of a single word. This is obviously faster than BST. 
        This is also faster than Hashing because of the ways it is implemented. We do not need to compute any hash function. No collision handling 
        is required (like we do in open addressing and separate chaining)

    * Another advantage of Trie is, we can easily print all words in alphabetical order which is not easily possible with hashing.

    * We can efficiently do prefix search (or auto-complete) with Trie.

Issues with Trie
    Faster but require HUGE memory for storing the strings



NOTE: Trie node class
struct TrieNode
{
     struct TrieNode *children[ALPHABET_SIZE];

     // isEndOfWord is true if the node
     // represents end of a word
     bool isEndOfWord;
};

'''

class TrieNode:
      
    # Trie node class
    def __init__(self):
        self.children = [None]*26
  
        # isEndOfWord is True if node represent the end of the word
        self.isEndOfWord = False