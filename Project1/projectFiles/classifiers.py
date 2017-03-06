from __future__ import division
from sklearn import datasets
import numpy
import sys
import preprocessSentences
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import sklearn.naive_bayes as skNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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

#############################################################################

# predicts the result of the test data
def classify_test_data(classifier,bowTest):
    pred_test = classifier.predict(bowTest)
    pred_prob_test = classifier.predict_proba(bowTest)
    return(pred_test,pred_prob_test)



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
    return(gnb)

# Bernoulli Naive Bayes classifier
def bernoulliNB(bagofwords,results):
    gnb = skNB.BernoulliNB()
    gnb.fit(bagofwords,results)
    return(gnb)

# SVM classifier
def SVM(bagofwords,results):
    gnb = svm.SVC()
    gnb.fit(bagofwords,results)
    return(gnb)

# DecisionTreeClassifier classifier
def decisionTree(bagofwords,results):
    gnb = tree.DecisionTreeClassifier()
    gnb.fit(bagofwords,results)
    return(gnb)

# RandomForestClassifier classifier
def randomForest(bagofwords,results):
    gnb = RandomForestClassifier(n_estimators=14)
    gnb.fit(bagofwords,results)
    return(gnb)

# K-nearest neighbours classifier
def kNearestNeighbours(bagofwords, results):
    gnb = KNeighborsClassifier(n_neighbors=5)
    gnb.fit(bagofwords, results)
    return(gnb)

#############################################################################
# Classifier metrics
#############################################################################

def classifier_metrics( resultsTest, classified_test_data):


    # score = gaussianNB_classifier.(bowTest, resultsTest)

    # print "score: " + str(score)

    # print "resultsTest: " + str(resultsTest)
    # print "classified_test_data: " + str(classified_test_data)

    tp = true_positive(resultsTest, classified_test_data)
    fn = false_negative(resultsTest, classified_test_data)
    tn = true_negative(resultsTest, classified_test_data)
    fp = false_positive(resultsTest, classified_test_data)

    # print "tp: " + str(tp)
    # print "fn: " + str(fn)
    # print "tn: " + str(tn)
    # print "fp: " + str(fp)

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

    # fpr, tpr, thresholds = metrics.roc_curve(classified_test_data, classified_test_data_prob, pos_label=1)

    # print "fpr: "  + str(fpr)
    # print "tpr: " + str(tpr)
    # print "thresholds: " + str(thresholds)
    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.plot(fpr, tpr, '-o')
    # plt.subplot(212)
    # plt.plot(thresholds,fpr, '-o' )
    # plt.show()
    # print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))

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
    # print "classes: " + str(classes)

    final_test_results = numpy.zeros(len(classes))
    test_multinomialNB_k_fold(bow, classes, final_test_results, kf)
    test_bernoulliNB_k_fold(bow, classes, final_test_results, kf)
    # test_gaussianNB_k_fold(bow, classes, final_test_results, kf)
    test_SVM_k_fold(bow, classes, final_test_results, kf)
    test_decisionTree_k_fold(bow, classes, final_test_results, kf)
    test_randomForest_k_fold(bow, classes, final_test_results, kf)
    test_kNearestNeighbours_k_fold(bow, classes, final_test_results, kf)

def kFold_hyperparameter_fitting():
    bow, results = prepare_training_data()
    final_test_results = numpy.zeros(len(results))
    kf = KFold(n_splits=10)
    kf.get_n_splits(bow)
    fit_hyperparameter_knearest(bow, results, final_test_results, kf)
    # fit_hyperparameter_forest(bow, results, final_test_results, kf)

#############################################################################
# Test Classifiers(not-k-fold)
#############################################################################

def test_gaussianNB(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = gaussianNB(bagofwords, results)
    classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob]
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold gaussian classifier: " + str(time_taken)
    print "*******************************************************"

def test_multinomialNB(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = multinomialNB(bagofwords, results)
    classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob]
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold multinomial classifier: " + str(time_taken)
    print "*******************************************************"

def test_bernoulliNB(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = bernoulliNB(bagofwords, results)
    classified_test_data, classified_test_data_prob= classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob]
    # print "classified_test_data_prob: " + str(classified_test_data_prob)
    # print "classified_test_data_prob: " + str(sorted(classified_test_data_prob, reverse=True))
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold bernoulli classifier: " + str(time_taken)
    print "*******************************************************"

def test_SVM(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = SVM(bagofwords, results)
    classified_test_data = gaussianNB_classifier.predict(bowTest)
    time_taken = time.time() - start_time

    # classified_test_data_prob = [max(item) for item in classified_test_data_prob]
    classifier_metrics(resultsTest, classified_test_data, classified_test_data)
    print "time_taken by non-k-fold SVM classifier: " + str(time_taken)
    print "*******************************************************"

def test_decisionTree(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = decisionTree(bagofwords, results)
    classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob ]

    # print "classified_test_data_prob: " + str(classified_test_data_prob)
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold decisionTree classifier: " + str(time_taken)
    print "*******************************************************"

def test_randomForest(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = randomForest(bagofwords, results)
    classified_test_data, classified_test_data_prob= classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob]
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold randomForest classifier: " + str(time_taken)
    print "*******************************************************"

def test_kNearestNeighbours(bagofwords,results, resultsTest, bowTest):
    start_time = time.time()
    gaussianNB_classifier = kNearestNeighbours(bagofwords, results)
    classified_test_data, classified_test_data_prob= classify_test_data(gaussianNB_classifier, bowTest)
    time_taken = time.time() - start_time
    # print "classified_test_data: " + str(classified_test_data)
    # print "classified_test_data_prob_prev: " + str(classified_test_data_prob.tolist())
    classified_test_data_prob = [item[1] for item in classified_test_data_prob]
    classifier_metrics(resultsTest, classified_test_data, classified_test_data_prob)
    print "time_taken by non-k-fold kNearestNeighbours classifier: " + str(time_taken)
    print "*******************************************************"

#############################################################################
# Test Classifiers(k-fold)
#############################################################################

def test_gaussianNB_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = gaussianNB(bow_train, results_train)
        classified_test_data,classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold gaussian classifier: " + str(total_time)
    print "*******************************************************"

def test_multinomialNB_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = multinomialNB(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold multinomial classifier: " + str(total_time)
    print "*******************************************************"

def test_bernoulliNB_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = bernoulliNB(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold bernoulli classifier: " + str(total_time)
    print "*******************************************************"

def test_SVM_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = decisionTree(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold SVM classifier: " + str(total_time)
    print "*******************************************************"

def test_decisionTree_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = decisionTree(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold decisionTree classifier: " + str(total_time)
    print "*******************************************************"

def test_randomForest_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = decisionTree(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold randomForest classifier: " + str(total_time)
    print "*******************************************************"

def test_kNearestNeighbours_k_fold(bow, classes, final_test_results, kf):
    start_time = time.time()
    for train_index, test_index in kf.split(bow):
        # print("TRAIN:", train_index, "TEST:", test_index)
        bow_train, bow_test = bow[train_index], bow[test_index]
        results_train, results_test = classes[train_index], classes[test_index]
        # print("bow_train: ", bow_train, "bow_test: ", bow_test)
        # print("results_train: ", results_train, "results_test: ", results_test)
        gaussianNB_classifier = kNearestNeighbours(bow_train, results_train)
        classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
        for index, elem in enumerate(test_index):
            final_test_results[elem] = classified_test_data[index]
    total_time = time.time() - start_time
    classifier_metrics(classes, final_test_results)
    print "time_taken by k-fold kNearestNeighbours classifier: " + str(total_time)
    print "*******************************************************"

#############################################################################
# Fit hyperparameter
#############################################################################
def fit_hyperparameter_knearest(bow, classes, final_test_results, kf):
    accu_arr = numpy.zeros(10)
    for ind, cnt in enumerate(range(11,21)):
        start_time = time.time()
        for train_index, test_index in kf.split(bow):
            # print("TRAIN:", train_index, "TEST:", test_index)
            bow_train, bow_test = bow[train_index], bow[test_index]
            results_train, results_test = classes[train_index], classes[test_index]
            # print("bow_train: ", bow_train, "bow_test: ", bow_test)
            # print("results_train: ", results_train, "results_test: ", results_test)
            # gaussianNB_classifier = kNearestNeighbours(bow_train, results_train)
            gaussianNB_classifier = KNeighborsClassifier(n_neighbors=cnt)
            gaussianNB_classifier.fit(bow_train, results_train)
            classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
            for index, elem in enumerate(test_index):
                final_test_results[elem] = classified_test_data[index]
        total_time = time.time() - start_time
        accu_arr[ind] = accuracy(classes, final_test_results)
        final_test_results = numpy.zeros(len(final_test_results))
    print "accu_arr: " + str(accu_arr)
    import matplotlib.pyplot as plt
    plt.plot(range(11,21), accu_arr, '-o')
    plt.title("Accuracy vs. hyperparameter values for k-nearest neighbours classifier")
    plt.xlabel("Hyperparameter values")
    plt.ylabel("Accuracy")
    plt.show()
    print "best knearest hyperparameter is: " + str(max(accu_arr))
    print "time_taken by k-fold kNearestNeighbours classifier: " + str(total_time)
    print "*******************************************************"
def fit_hyperparameter_forest(bow, classes, final_test_results, kf):
    accu_arr = numpy.zeros(31)
    for ind, cnt in enumerate(range(50,81)):
        start_time = time.time()
        for train_index, test_index in kf.split(bow):
            # print("TRAIN:", train_index, "TEST:", test_index)
            bow_train, bow_test = bow[train_index], bow[test_index]
            results_train, results_test = classes[train_index], classes[test_index]
            # print("bow_train: ", bow_train, "bow_test: ", bow_test)
            # print("results_train: ", results_train, "results_test: ", results_test)
            # gaussianNB_classifier = kNearestNeighbours(bow_train, results_train)
            gaussianNB_classifier = RandomForestClassifier(n_estimators=cnt)
            gaussianNB_classifier.fit(bow_train, results_train)
            classified_test_data, classified_test_data_prob = classify_test_data(gaussianNB_classifier, bow_test)
            for index, elem in enumerate(test_index):
                final_test_results[elem] = classified_test_data[index]
        total_time = time.time() - start_time
        accu_arr[ind] = accuracy(classes, final_test_results)
        final_test_results = numpy.zeros(len(final_test_results))
    print "accu_arr: " + str(accu_arr)
    import matplotlib.pyplot as plt
    plt.plot(range(50,81), accu_arr, '-o')
    plt.title("Accuracy vs. hyperparameter values for random forest classifier")
    plt.xlabel("Hyperparameter values")
    plt.ylabel("Accuracy")
    plt.show()
    print "best knearest hyperparameter is: " + str(max(accu_arr))
    print "time_taken by k-fold randomforest classifier: " + str(total_time)
    print "*******************************************************"

#############################################################################
#main function
def main(argv):
    (bagofwords,results) = prepare_training_data()
    (resultsTest, bowTest)=prepare_test_data()
    resultsTest = map(int, resultsTest)


    # test_gaussianNB(bagofwords,results, resultsTest, bowTest)
    # test_multinomialNB(bagofwords,results, resultsTest, bowTest)
    # test_bernoulliNB(bagofwords, results, resultsTest, bowTest)
    # test_SVM(bagofwords,results, resultsTest, bowTest)
    # test_decisionTree(bagofwords, results, resultsTest, bowTest)
    # test_randomForest(bagofwords, results, resultsTest, bowTest)
    # test_kNearestNeighbours(bagofwords,results, resultsTest, bowTest)

    kFold()

    return()


if __name__ == "__main__":
  main(sys.argv[1:])
    # kFold_hyperparameter_fitting()