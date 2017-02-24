from sklearn import datasets
import numpy
import preprocessSentences
from sklearn.naive_bayes import GaussianNB

# predicts the result of the test data
def classify_test_data(classifier):
    vocab = numpy.genfromtxt('./data/out_vocab_5.txt', delimiter='', dtype=None)
    (docsTest, classesTest, samplesTest) = preprocessSentences.tokenize_corpus('./data/test.txt', False)
    bowtest = preprocessSentences.find_wordcounts(docsTest, vocab)
    print bowtest

    pred_test = classifier.predict(bowtest)
    print pred_test.tolist()

#prepares the arrays needed to train a classifier
def prepare_to_classify():
    bagofwords = numpy.genfromtxt('./data/out_bag_of_words_5.csv', delimiter=',')
    print bagofwords
    results = numpy.genfromtxt('./data/out_classes_5.txt', delimiter='', dtype=None)
    print results.tolist()
    return(bagofwords,results)

# Gaussian Naive Bayes classifier
def gaussianNB(bagofwords,results):
    gnb = GaussianNB()
    gnb.fit(bagofwords,results)
    pred_train = gnb.predict(bagofwords)
    print pred_train.tolist()
    return(gnb)


# Multinomial Naive Bayes classifier
