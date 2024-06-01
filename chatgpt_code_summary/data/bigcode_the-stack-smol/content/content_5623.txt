import sys
##print ("This is the name of the script: ", sys.argv[0])
##print ("Number of arguments: ", len(sys.argv))
##print ("The arguments are: " , str(sys.argv))

lemmas = []
lemmas_cleaned = []
nums = ['1','2','3','4','5','6','7','8','9','0']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','k','l','m','n','o',
            'p','q','r','s','t','u','v','w','x','y','z',
            'ā','â','ā',
            'ê','ė', 'ē',
            'ô','ō',
            'ū','û',
            'ī','î']

POS_tags = ['as., st. V. (1)', 'as., st. V. (2)', 'as., st. V. (3a)', 'as., st. V. (3b)', 'as., st. V. (4)', 'as., st. V. (5)', 'as., st. V. (6)', 'as., st. V. (7)', \
'as., sw. V. (1a)', 'as., sw. V. (1b)', 'as., sw. V. (2)', \
'as., red. V.',\
            'as., st. M. (a)']

def dict_scrape(POS, dictionaryfile='as_freq.txt'):
    """Scrapes a dictionary for a given part of speech. POS tags in POS_tags.

    POS(str), dictionaryfile(str-of-filename) -> list-of-strings
    """
    if POS in POS_tags:
        with open(dictionaryfile) as to_scrape:
            for line in to_scrape:
                if POS in line:
                    lemmas.append(line)
    for line in lemmas:
#1,                                       bāga, as., st. F. (ō)?, sw. F. (n)?: nhd. Streit
        i=0
        for char in line[:44]:
            if char not in alphabet:
                i=i+1
        lemmas_cleaned.append(line[i:].strip().replace('*','').replace('?','')+'\n')
        #scrub line of the frequency data, begin with headword?

##    print("Found " + str(len(lemmas_cleaned)) + " lemmas matching that category")
    return lemmas_cleaned   


dict_scrape(sys.argv[1])
for line in lemmas_cleaned:
    with open(sys.argv[1][5:],'w+') as to_write:
        to_write.write(line)
