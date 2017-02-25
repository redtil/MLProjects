from __future__ import division
from sklearn import datasets
import numpy
import sys
import preprocessSentences
import sklearn.naive_bayes as skNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import time

#############################################################################
# Metrics used to compare classifiers
#############################################################################
def true_positive(resultsTest, classified_test_data):
    count = 0
    for r, c in zip(resultsTest, classified_test_data):
        if((r == 1) and (c == 1)):
            count = count + 1
    return count
def true_negative(resultsTest, classified_test_data):
    count = 0
    for r, c in zip(resultsTest, classified_test_data):
        if ((r == 0) and (c == 0)):
            count = count + 1
    return count

def false_positive(resultsTest, classified_test_data):
    count = 0
    for r, c in zip(resultsTest, classified_test_data):
        if ((r == 0) and (c == 1)):
            count = count + 1
    return count

def false_negative(resultsTest, classified_test_data):
    count = 0
    for r, c in zip(resultsTest, classified_test_data):
        if ((r == 1) and (c == 0)):
            count = count + 1
    return count

def accuracy(resultsTest, classified_test_data):
    count = 0
    for r, c in zip(resultsTest, classified_test_data):
        if(r == c):
            count = count + 1
    return count/len(resultsTest)

def precision(tp, fp):
    precision = tp /(tp+fp)
    return(precision)

def recall(tp,fn):
    recall = tp/(tp+fn)
    return(recall)

def f1Score(precision, recall):
    f1_score = 2*((precision*recall)/(precision + recall))
    return(f1_score)


##############################################################################
# Preparation of data to be classified
#############################################################################


#prepares the arrays needed to train a classifier
def prepare_training_data():
    bagofwords = numpy.genfromtxt('./data/out_bag_of_words_5.csv', delimiter=',')
    results = numpy.genfromtxt('./data/out_classes_5.txt', delimiter='', dtype=None)
    return(bagofwords,results)

def prepare_test_data():
    vocab = numpy.genfromtxt('./data/out_vocab_5.txt', delimiter='', dtype=None)
    (docsTest, classesTest, samplesTest) = preprocessSentences.tokenize_corpus('./data/test.txt', False)
    bowTest = preprocessSentences.find_wordcounts(docsTest, vocab)
    return(classesTest, bowTest)

def kFold():
    (docs, classes, samples, words) = preprocessSentences.tokenize_corpus('./data/train_test_together.txt', train=True)
    classes = map(int, classes)
    classes = numpy.asarray(classes)
    vocab = preprocessSentences.wordcount_filter(words, num=5)
    bow = preprocessSentences.find_wordcounts(docs, vocab)
    kf = KFold(n_splits=10)
    kf.get_n_splits(bow)
    print "classes: " + str(classes)

    final_test_results = numpy.zeros(len(classes))
    for train_index, test_index in kf.split(bow):
        print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        print("bow_train: ", bow_train, "bow_test: ", bow_test)
        print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = gaussianNB(bow_train, results_train)
        classified_test_data = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    metrics_classifier(classes, final_test_results)




#############################################################################

# predicts the result of the test data
def classify_test_data(classifier,bowTest):
    pred_test = classifier.predict(bowTest)
    return(pred_test)


#############################################################################
# Various classifiers
#############################################################################
# Gaussian Naive Bayes classifier
def gaussianNB(bagofwords,results):
    gnb = skNB.GaussianNB()
    gnb.fit(bagofwords,results)
    return(gnb)


# Multinomial Naive Bayes classifier
def multinomialNB(bagofwords,results):
    gnb = skNB.MultinomialNB()
    gnb.fit(bagofwords,results)
    pred_train = gnb.predict(bagofwords)
    return(gnb)

#############################################################################
# Classifier metrics
#############################################################################

def metrics_classifier( resultsTest, classified_test_data):


    # score = gaussianNB_classifier.(bowTest, resultsTest)

    # print "score: " + str(score)

    print "resultsTest: " + str(resultsTest)
    print "classified_test_data: " + str(classified_test_data)

    tp = true_positive(resultsTest, classified_test_data)
    fn = false_negative(resultsTest, classified_test_data)
    tn = true_negative(resultsTest, classified_test_data)
    fp = false_positive(resultsTest, classified_test_data)

    print "tp: " + str(tp)
    print "fn: " + str(fn)
    print "tn: " + str(tn)
    print "fp: " + str(fp)

    prec = precision(tp, fp)
    rec = recall(tp, fn)
    print "prec:  " + str(prec)
    print "recall: " + str(rec)
    f1 = f1Score(prec, rec)
    print "f1-score: " + str(f1)
    acc = accuracy(resultsTest, classified_test_data)
    acc2 = accuracy_score(resultsTest, classified_test_data)
    print "accuracy: " + str(acc)
    print "accuracy2: " + str(acc2)

#############################################################################

#main function
def main(argv):
    (bagofwords,results) = prepare_training_data()
    (resultsTest, bowTest)=prepare_test_data()
    resultsTest = map(int, resultsTest)
    start_time = time.time()
    gaussianNB_classifier = gaussianNB(bagofwords, results)
    classified_test_data = classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    print "time taken: " + str(time_taken)

    metrics_classifier(resultsTest,classified_test_data)
    return()


if __name__ == "__main__":
    kFold()
  # main(sys.argv[1:])