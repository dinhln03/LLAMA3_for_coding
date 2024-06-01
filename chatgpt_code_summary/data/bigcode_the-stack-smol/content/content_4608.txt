import re
import numpy
import math
import sys

#implementing the stop words and 
def extractCleanWords(review):
    stopWords = ["in", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now"]
    words = re.sub("[^\w]", " ", review).split()
    cleanWords = [i.lower() for i in words if i not in stopWords]
    return cleanWords

#used by bag of words to create the vocab dictionary
def createVocabTokens(reviews):
    vocab = []
    for review in reviews:
        token = extractCleanWords(review)
        vocab.extend(token)
    vocab = sorted(list(set(vocab)))
    return vocab

"""the bag of words for multinomialNB does not need to create
    matrixes for each review because it takes too much space and slows
    it down and it is not neccesary. The bag of words returns a 
    dictionary with the frequencies for each word used for the numerator of
    P(xi|ci), the total words in the classifier used for the denom of
    P(xi|ci), and the number of reviews for the class used to calculate 
    the prior probabilities for each class"""

def bagOfWords_MultinomialNB(txtFile):
    total_words = 0
    reviewFile = txtFile
    with open(reviewFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    #print(len(reviewList))
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    vocabTokens = createVocabTokens(reviewList)
    #print("Word bank for reviews: \n{0} \n".format(vocabTokens));
    #print(len(vocabTokens))
    #bagOfWords(reviewFile)
    numReviews = len(reviewList)
    #print(len(reviewList))
    #print(len(vocabTokens))
    vocabDict = dict.fromkeys(vocabTokens, 0)
    #matrix = numpy.zeros(shape = (len(reviewList),len(vocabTokens)))
    for i in range(len(reviewList)):
        words = extractCleanWords(reviewList[i])
        #bagList = numpy.zeros(len(vocabTokens))
        for word in words:
            vocabDict[word] += 1
            total_words +=1
            #if word in vocabTokens:
                #bagList[vocabTokens.index(word)] +=1
        #print(i, " out of ", len(vocabTokens), " done")
        #matrix[i] = bagList
        #print("{0}\n{1}\n".format(review,numpy.array(bagList)))
    return vocabDict, total_words, numReviews

def bagOfWords_GaussianNB(txtFile):
    total_words = 0
    reviewFile = txtFile
    with open(reviewFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    #print(len(reviewList))
    numReviews = len(reviewList)
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    vocabTokens = createVocabTokens(reviewList)
    vocabDict = dict.fromkeys(vocabTokens, 0)
    for i in range(len(reviewList)):
        words = extractCleanWords(reviewList[i])
        #bagList = numpy.zeros(len(vocabTokens))
        for word in words:
            vocabDict[word] += 1
    sparseMatrix = []
    for i in range(len(reviewList)):
        #print("Gauss: ", i)
        words = extractCleanWords(reviewList[i])
        bagList = {}
        for word in words:
            if word in bagList:
                bagList[word] +=1
            else:
                bagList[word] = 1
        sparseMatrix.append(bagList)
    return sparseMatrix, vocabDict, numReviews

#calculates the mean and varience using bag of words 
def calcMean_Var(txtFile, tfidforBOW):
    if tfidforBOW == 1:#using bag of words
        sparseMatrix, vocabDict, numReviews = bagOfWords_GaussianNB(txtFile)
    else:
        sparseMatrix, vocabDict, numReviews = tf_idf(txtFile)
    meanVarDict = {}
    meanVarTouple = [0,0]
    for word in vocabDict:
        meanVarTouple[0] = (vocabDict[word] / numReviews)
        #print(meanVarTouple[0])
        var = 0
        for m in sparseMatrix:
            if word in m:
                var += ((m[word]-meanVarTouple[0])**2)
            else:
                var += ((-1*meanVarTouple[0])**2)
        meanVarTouple[1] = (var / (numReviews -1))
        meanVarDict[word] = meanVarTouple
        #print("Gauss: ", meanVarTouple)
    return meanVarDict
    
def gaussian_BOW(trainDataPos, trainDataNeg, testData, c):
    testFile = testData
    with open(testFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    #prediction will be used for the accuracy of the classifier
    prediction = []
    meanVarDictPOS = calcMean_Var(trainDataPos,1)
    meanVarDictNEG = calcMean_Var(trainDataNeg,1)
    testWordFreq = {}
    for review in reviewList:
        wordsInReview = extractCleanWords(review)
        for word in wordsInReview:
            if (word in meanVarDictPOS) or (word in meanVarDictNEG):
                if word in testWordFreq:
                    testWordFreq[word] += 1
                else:
                    testWordFreq[word] = 1
    for review in reviewList:
        wordsInReview = list(set(extractCleanWords(review)))
        probPos =0
        probNeg =0
        for word in wordsInReview:
            if word in meanVarDictPOS:
                probPos += (math.log((1/(math.sqrt(2*math.pi*meanVarDictPOS[word][1])))) - (((testWordFreq[word] - meanVarDictPOS[word][0])**2)/((meanVarDictPOS[word][1]**2))))
            if word in meanVarDictNEG:
                probNeg += (math.log((1/(math.sqrt(2*math.pi*meanVarDictNEG[word][1])))) - (((testWordFreq[word] - meanVarDictNEG[word][0])**2)/((meanVarDictNEG[word][1]**2))))
        if probPos > probNeg:
            prediction.append(1)
        else:
            prediction.append(0)
    poss = 0
    for p in prediction:
        if p == c:
            poss +=1
    return(poss/len(prediction))


def tf_idf(txtFile):
    reviewFile = txtFile
    with open(reviewFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    #print(len(reviewList))
    numReviews = len(reviewList)
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    vocabTokens = createVocabTokens(reviewList)
    vocabDictIDF = dict.fromkeys(vocabTokens, 0)
    """for i in range(len(reviewList)):
        words = extractCleanWords(reviewList[i])
        #bagList = numpy.zeros(len(vocabTokens))
        for word in words:
            vocabDict[word] += 1"""
    totalNumWords = 0
    sparseMatrixTFIDF = []

    for i in range(len(reviewList)):
        #print("TFidf: ", i)
        words = extractCleanWords(reviewList[i])
        bagListTF = {}
        for word in words:
            totalNumWords +=1
            if word in bagListTF:
                bagListTF[word] +=1
            else:
                bagListTF[word] = 1
        for word in list(set(words)):
            bagListTF[word] = (bagListTF[word]/totalNumWords)
            vocabDictIDF[word]+=1
        sparseMatrixTFIDF.append(bagListTF)
        #print(i)
    
    #using the tf vlues in the sparse matrix and idf values in 
    #the vocab dict we can get the tf idf and hold it in sparse matrix
    vocabDict = dict.fromkeys(vocabTokens, 0)
    for dictTF in sparseMatrixTFIDF:
        for word in dictTF:
            dictTF[word] = (dictTF[word] * (math.log((len(reviewList)/vocabDictIDF[word]))))
            vocabDict[word]+= dictTF[word]
    #print(sparseMatrixTFIDF)
    return sparseMatrixTFIDF, vocabDict, numReviews
    


def gaussian_tf_idf(trainDataPos, trainDataNeg, testData, c):
    testFile = testData
    with open(testFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    #prediction will be used for the accuracy of the classifier
    prediction = []
    meanVarDictPOS = calcMean_Var(trainDataPos,0)
    meanVarDictNEG = calcMean_Var(trainDataNeg,0)
    testSparseTFIDF, testVocabDict, testNumReviews = tf_idf(testData)

    for review in reviewList:
        wordsInReview = list(set(extractCleanWords(review)))
        probPos =0
        probNeg =0
        for word in wordsInReview:
            if word in meanVarDictPOS:
                probPos += (math.log((1/(math.sqrt(2*math.pi*meanVarDictPOS[word][1])))) - (((testVocabDict[word] - meanVarDictPOS[word][0])**2)/(2*(meanVarDictPOS[word][1]**2))))
            if word in meanVarDictNEG:
                probNeg += (math.log((1/(math.sqrt(2*math.pi*meanVarDictNEG[word][1])))) - (((testVocabDict[word] - meanVarDictNEG[word][0])**2)/(2*(meanVarDictNEG[word][1]**2))))
        if probPos > probNeg:
            prediction.append(1)
        else:
            prediction.append(0)
    poss = 0
    for p in prediction:
        if p == c:
            poss +=1
    return(poss/len(prediction))




def multinomialNB(trainDataPos, trainDataNeg, testData, c):

    testFile = testData
    with open(testFile, 'r', encoding='utf8') as file:
        txt = file.read().replace('\n','')
    reviewList = txt.split("<br /><br />")
    for review in reviewList:
        review = re.sub("[^\w\s]", "", review)
    #prediction will be used for the accuracy of the classifier
    prediction = []

    #getting the dict, word count and review count for pos and neg from BOW
    posDict, posWordCount, posdocs = bagOfWords_MultinomialNB(trainDataPos)
    negDict, negWordCount, negdocs = bagOfWords_MultinomialNB(trainDataNeg)
    
    """TEST PRINT STATEMENTS
    print("Pos dic: ", len(posDict))
    print("Neg dic: ", len(negDict))
    print("Pos word count: ", posWordCount)
    print("Neg word count: ", negWordCount)
    print("Pos docs: ", posdocs)
    print("Neg docs: ", negdocs)"""

    #alpha is the smoothing paramater, through trial i found that a value
    #of 18 will have the highest prediction frequency
    alpha = 18

    #calculating the prior log prob for pos and neg
    priorLogPosProb =math.log( posdocs / (negdocs + posdocs))
    priorLogNegProb =math.log( negdocs / (negdocs + posdocs))

    """for each review in our test, we extract the words and calculate 
    the log prob for that word given pos and neg and add this with 
    the prior log probability, then we compare the pos and neg total
    probabilities and assign a 1 if the pos > neg, and 0 for the opposite
    We check the prediction list and calculate the accurace for the 
    given classifier"""

    for review in reviewList:
        wordsInReview = list(set(extractCleanWords(review)))
        logProbPos = 0
        logProbNeg = 0
        posPercent = 0
        negPercent = 0
        for word in wordsInReview:
            if word not in posDict:
                logProbPos += math.log( ((alpha) / (posWordCount+(alpha*len(posDict) ) ) ) )
            if word in posDict:
                logProbPos += math.log( ((posDict[word] + alpha) / (posWordCount+(alpha*len(posDict) ) ) ) )
            if word not in negDict:
                logProbNeg += math.log( ((alpha) / (negWordCount+(alpha*len(negDict) ) ) ) )
            if word in negDict:
                logProbNeg += math.log( ((negDict[word] + alpha) / (negWordCount+(alpha*len(negDict) ) ) ) )
        posPercent = priorLogPosProb + logProbPos
        negPercent = priorLogNegProb + logProbNeg

        if posPercent > negPercent:
            prediction.append(1)
        else:
            prediction.append(0)
    poss = 0
    for p in prediction:
        if p == c:
            poss +=1
    return(poss/len(prediction))

#setting the arguments
train_pos = sys.argv[1]
train_neg = sys.argv[2]
test_pos = sys.argv[3]
test_neg = sys.argv[4]
#getting the accuracy for multinomial for pos test and neg test
posAcc = multinomialNB(train_pos, train_neg, test_pos,1)  
negAcc = multinomialNB(train_pos, train_neg, test_neg,0) 

#calculating the average accuracy and printing it out
multinomialAcc = (posAcc+negAcc) / 2
print("MultinomialNB with bag of words accuracy: ", multinomialAcc)

gposAcc = gaussian_BOW(train_pos, train_neg, test_pos,1)
gnegAcc = gaussian_BOW(train_pos, train_neg, test_neg,0)
gaussAcc = (gposAcc+gnegAcc) / 2

print("Gaussian with bag of words accuracy: ", gaussAcc)

#calcMean_Var(train_pos,1)
#tf_idf(train_pos)
tposAcc = gaussian_tf_idf(train_pos, train_neg, test_pos,1)
tnegAcc = gaussian_tf_idf(train_pos, train_neg, test_neg,0)
tgaussAcc = (tposAcc+tnegAcc) / 2

print("Gaussian with tf_idf acc: ", tgaussAcc)
