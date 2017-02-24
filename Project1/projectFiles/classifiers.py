from sklearn import datasets
import numpy
import sys
import preprocessSentences
import sklearn.naive_bayes as skNB

def precision(tp, fp):
    precision = tp /(tp+fp)
    return(precision)

def recall(tp,fn):
    recall = tp/(tp+fn)
    return(recall)

def f1Score(precision, recall):
    f1_score = 2*((precision*recall)/(precision + recall))
    return(f1_score)

def prepare_to_classify_test_data():
    vocab = numpy.genfromtxt('./data/out_vocab_5.txt', delimiter='', dtype=None)
    (docsTest, classesTest, samplesTest) = preprocessSentences.tokenize_corpus('./data/test.txt', False)
    bowTest = preprocessSentences.find_wordcounts(docsTest, vocab)
    print bowTest
    return(classesTest, bowTest)

# predicts the result of the test data
def classify_test_data(classifier,bowTest):
    pred_test = classifier.predict(bowTest)
    print pred_test.tolist()
    return(pred_test)

#prepares the arrays needed to train a classifier
def prepare_training_data():
    bagofwords = numpy.genfromtxt('./data/out_bag_of_words_5.csv', delimiter=',')
    print bagofwords
    results = numpy.genfromtxt('./data/out_classes_5.txt', delimiter='', dtype=None)
    print results.tolist()
    return(bagofwords,results)

# Gaussian Naive Bayes classifier
def gaussianNB(bagofwords,results):
    gnb = skNB.GaussianNB()
    gnb.fit(bagofwords,results)
    pred_train = gnb.predict(bagofwords)
    print pred_train.tolist()
    return(gnb)


# Multinomial Naive Bayes classifier
def multinomialNB(bagofwords,results):
    gnb = skNB.MultinomialNB()
    gnb.fit(bagofwords,results)
    pred_train = gnb.predict(bagofwords)
    print pred_train.tolist()
    return(gnb)

def main(argv):
    (bagofwords,results) = prepare_training_data()
    gaussianNB_classifier = gaussianNB(bagofwords,results)
    (classesTest, bowTest)=prepare_to_classify_test_data()
    classified_test_data = classify_test_data(gaussianNB_classifier,bowTest)

    return()


if __name__ == "__main__":
  main(sys.argv[1:])