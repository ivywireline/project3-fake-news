import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import cPickle as pickle


from collections import defaultdict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Uncomment the following line after installing GraphViz if need to generate part 7 visualization
# import graphviz

real_filename = 'clean_real.txt'
fake_filename = 'clean_fake.txt'


print_verbose = True


class NaiveBayesModel:
    def __init__(self, **kwargs):
        self.probs_real = kwargs.pop('probs_real', None)
        self.probs_fake = kwargs.pop('probs_fake', None)
        self.m = kwargs.pop('m', 1)
        self.p = kwargs.pop('p', 0.1)
        self.num_real = kwargs.pop('num_real_headlines', 0)
        self.num_fake = kwargs.pop('num_fake_headlines', 0)
        self.word_list = kwargs.pop('word_list', list())
        self.real_word_counts = kwargs.pop('real_word_counts', dict())
        self.fake_word_counts = kwargs.pop('fake_word_counts', dict())

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)


def get_wordlist(*args):
    """
    Returns a list containing all of the unique words in all lists of
    headlines passed into this function.
    This list is sorted alphabetically

    *args: Lists of headlines as args

    >>> get_wordlist(['headline one is great', 'headline two is great'], ['headline three is great'])
    {'headline', 'one', 'is', 'great', 'two', 'three'}
    """
    headlines = [hl for hls in args for hl in hls]
    word_list = set()

    for hl in headlines:
        for w in hl.split(' '):
            word_list.add(w)

    return sorted(list(word_list))


def load_headlines(headlines_filename):
    with open(headlines_filename, 'r') as infile:
        headlines = [hl.strip() for hl in infile.readlines() if hl.strip()]

    random.shuffle(headlines)

    num_validation = int(len(headlines) * 0.15)
    num_test = int(len(headlines) * 0.15)
    num_testvalidation = num_validation + num_test

    validation = headlines[:num_validation]
    test = headlines[num_validation: num_testvalidation]
    training = headlines[num_testvalidation:]

    return training, validation, test


def count_word_occurrance(headlines):
    '''
    Counts how many headlines each word occurs in
    '''
    word_counts = defaultdict(lambda: 0)
    for line in headlines:
        already_found = set()
        for word in set(line.split(' ')):
            if word not in already_found:
                word_counts[word] += 1
                already_found.add(word)

    return word_counts

#################### Begin Part 2 ############################
def train_model(real_headlines, fake_headlines, m, p):
    word_list = get_wordlist(real_headlines, fake_headlines)
    real_counts = count_word_occurrance(real_headlines)
    fake_counts = count_word_occurrance(fake_headlines)
    probabilities_real = {}
    probabilities_fake = {}
    for word in word_list:
        # if word in ENGLISH_STOP_WORDS: continue
        if word in real_counts:
            probabilities_real[word] = (real_counts[word] + m * p) / float(len(real_headlines) + m)
        else:
            probabilities_real[word] = (0 + m * p) / float(len(real_headlines) + m)

        if word in fake_counts:
            probabilities_fake[word] = (fake_counts[word] + m * p) / float(len(fake_headlines) + m)
        else:
            probabilities_fake[word] = (0 + m * p) / float(len(fake_headlines) + m)

    return NaiveBayesModel(
        probs_real=probabilities_real,
        probs_fake=probabilities_fake,
        m=m,
        p=p,
        num_real_headlines=len(real_headlines),
        num_fake_headlines=len(fake_headlines),
        word_list=word_list,
        real_word_counts = real_counts,
        fake_word_counts = fake_counts,
    )

def predict_model(model, headline):
    probabilities_real = model.probs_real
    probabilities_fake = model.probs_fake
    real_count = model.num_real
    fake_count = model.num_fake
    word_list = model.word_list

    logprob_real = 0.0
    logprob_fake = 0.0
    real_class_prob = float(real_count) / (real_count + fake_count)
    fake_class_prob = float(fake_count) / (real_count + fake_count)
    headline_split = headline.split(' ')
    for word in word_list:
        if word in headline_split:
            logprob_real += math.log(probabilities_real[word])
            logprob_fake += math.log(probabilities_fake[word])
        else:
            logprob_real += math.log(1 - probabilities_real[word])
            logprob_fake += math.log(1 - probabilities_fake[word])
    real_prob = math.exp(logprob_real) * real_class_prob
    fake_prob = math.exp(logprob_fake) * fake_class_prob
    return real_prob > fake_prob


def tune_model(real_training, fake_training, real_validation, fake_validation):
    performance_report = {}
    m = 1
    while m <= 20:
        p = 0.1
        while p <= 1:
            model = train_model(real_training, fake_training, m, p)
            performance = get_performance(model, real_validation, fake_validation)
            if print_verbose:
                print m, p, performance
            performance_report[(m, p)] = performance
            p += 0.1
        m += 1

    return performance_report


def get_performance(model, real, fake):
    correct = 0

    for hl in real:
        if predict_model(model, hl):
            correct += 1

    for hl in fake:
        if not predict_model(model, hl):
            correct += 1

    return float(correct) / (len(real) + len(fake))


def get_total_performance(model, real_training, fake_training, real_test, fake_test, real_validation, fake_validation):
    accurate_count_training = 0
    accurate_count_test = 0
    total_training = len(real_training) + len(fake_training)
    total_test = len(real_test) + len(fake_test)

    # For debugging purposes
    total_real_training = len(real_validation)
    total_fake_training = len(fake_validation)
    accurate_count_real_validation = 0
    accurate_count_fake_validation = 0

    for real_sample in real_training:
        if predict_model(model, real_sample):
            accurate_count_training += 1

    for fake_sample in fake_training:
        if not predict_model(model, fake_sample):
            accurate_count_training += 1

    for real_sample in real_test:
        if predict_model(model, real_sample):
            accurate_count_test += 1

    for fake_sample in fake_test:
        if not predict_model(model, fake_sample):
            accurate_count_test += 1

    for real_sample in real_validation:
        if predict_model(model, real_sample):
            accurate_count_real_validation += 1

    for fake_sample in fake_validation:
        if not predict_model(model, fake_sample):
            accurate_count_fake_validation += 1

    performance_training = accurate_count_training / float(total_training)
    performance_test = accurate_count_test / float(total_test)

    performance_validation_real = accurate_count_real_validation / float(total_real_training)
    performance_validation_fake = accurate_count_fake_validation / float(total_fake_training)
    performance_validation_total = (accurate_count_real_validation + accurate_count_fake_validation) / (float(total_real_training) + float(total_fake_training))

    return performance_training, performance_test, performance_validation_real, performance_validation_fake, performance_validation_total

############################# Part 3 ####################################


def get_top_bottom_word_weights(model, stop_words=list()):
    prob_real = float(model.num_real) / (model.num_real + model.num_fake)
    prob_fake = float(model.num_fake) / (model.num_real + model.num_fake)

    keys_real = {
        k: (model.probs_real[k] * prob_real) / (float(model.real_word_counts[k] + model.fake_word_counts[k] + model.m * model.p) / (model.num_real + model.num_fake + model.m))
        for k in model.probs_real if k not in stop_words
    }

    keys_fake = {
        k: (model.probs_fake[k] * prob_fake) / (float(model.real_word_counts[k] + model.fake_word_counts[k] + model.m * model.p) / (model.num_real + model.num_fake + model.m))
        for k in model.probs_fake if k not in stop_words
    }

    keys_real_not = {
        k: ((1 - model.probs_real[k]) * prob_real) / (1 - float(model.real_word_counts[k] + model.fake_word_counts[k] + model.m * model.p) / (model.num_real + model.num_fake + model.m))
        for k in model.probs_real if k not in stop_words
    }
    keys_fake_not = {
        k: ((1 - model.probs_fake[k]) * prob_fake) / (1 - float(model.real_word_counts[k] + model.fake_word_counts[k] + model.m * model.p) / (model.num_real + model.num_fake + model.m))
        for k in model.probs_fake if k not in stop_words
    }

    sorted_keys_real = sorted(keys_real, key=keys_real.__getitem__, reverse=True)
    sorted_keys_fake = sorted(keys_fake, key=keys_fake.__getitem__, reverse=True)

    sorted_keys_real_not = sorted(keys_real_not, key=keys_real_not.__getitem__, reverse=True)
    sorted_keys_fake_not = sorted(keys_fake_not, key=keys_fake_not.__getitem__, reverse=True)

    real_top_10 = sorted_keys_real[:10]
    real_bottom_10 = sorted_keys_real_not[:10]

    fake_top_10 = sorted_keys_fake[:10]
    fake_bottom_10 = sorted_keys_fake_not[:10]

    return real_top_10, real_bottom_10, fake_top_10, fake_bottom_10


############################# Logistic Regression #############################
def process_headlines(real_training, fake_training):
    # Get the unique words list. Its index is important for feature vector representation.
    # Key is a unique word string. Value is the index of the word in unique_words_set

    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)
    idx = 0
    unique_words_dict = {}

    for word in unique_words_set:
        unique_words_dict[word] = idx
        idx += 1

    return unique_words_dict

def vectorize_headlines(real, fake, unique_words_dict):
    # 1 means real. 0 means fake.
    unique_words_set = unique_words_dict.keys()
    unique_words_number = len(unique_words_set)

    # Number of features is the number of the unique words in the training set
    batch_xs = []
    # There are only 2 classes - real or fake
    batch_y_s = [1 for _ in xrange(len(real))] + [0 for _ in xrange(len(fake))]

    for headline in real:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in set(headline_words):
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 1
        batch_xs.append(vector_headline)

    for headline in fake:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in set(headline_words):
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 1
        batch_xs.append(vector_headline)

    batch_xs = np.vstack(batch_xs)

    return batch_xs, batch_y_s

def transform_elements(lst):
    for i in range(len(lst)):
        if lst[i] <= 0.5:
            lst[i] = 0
        else:
            lst[i] = 1


def evaluate_logreg_model(model, x, expected_y):
    y_pred = model.forward(x).data.numpy().flatten()
    transform_elements(y_pred)
    return np.mean(y_pred == expected_y)


def plot_learning_curve(learning_data, filename="Part4LearningCurve"):
    train_vals = learning_data["train"]
    validation_vals = learning_data["valid"]
    test_vals = learning_data["test"]

    x_axis = [i for i in range(len(train_vals))]

    fig = plt.figure()
    plt.plot(x_axis, train_vals, 'r-', label="Training Set")
    plt.plot(x_axis, validation_vals, 'y-', label="Validation Set")
    plt.plot(x_axis, test_vals, label="Test Set")

    plt.xlabel("Number of Epochs")
    plt.ylabel("Proportion of Correct Guesses")
    plt.title("Learning Curves")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


def part4(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict, reg_lambda=0):
    # 1 means real. 0 means fake.
    unique_words_number = len(unique_words_dict)

    train_x, train_y = vectorize_headlines(real_training, fake_training, unique_words_dict)
    valid_x, valid_y = vectorize_headlines(real_validation, fake_validation, unique_words_dict)
    test_x, test_y = vectorize_headlines(real_test, fake_test, unique_words_dict)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    # Hyper Parameters
    input_size = unique_words_number
    num_classes = 1
    num_iters = 650
    learning_rate = 0.0001
    #reg_lambda = 0

    # model = LogisticRegression(input_size, num_classes)
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, num_classes, bias=False),
        torch.nn.Sigmoid(),
    )
    torch.nn.init.xavier_uniform(model[0].weight)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    x_train = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    y_train = Variable(torch.from_numpy(np.vstack(train_y)), requires_grad=False).type(dtype_float)

    x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)

    x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    performance_data_training = []
    performance_data_valid = []
    performance_data_test = []

    for i in range(num_iters):
        # Forward + Backward + Optimize
        y_pred = model(x_train)

        optimizer.zero_grad()
        loss = loss_fn.forward(y_pred, y_train)

        loss.backward()
        optimizer.step()

        if print_verbose:
            print ('Epoch: [%d/%d], Loss: %.4f' % (i+1, num_iters, loss.data[0]))

        # evaluate model
        performance_data_training.append(evaluate_logreg_model(model, x_train, train_y))
        performance_data_valid.append(evaluate_logreg_model(model, x_valid, valid_y))
        performance_data_test.append(evaluate_logreg_model(model, x_test, test_y))


    # Make predictions using the training set
    train_perf = evaluate_logreg_model(model, x_train, train_y)
    print "Performance on Training Set", train_perf

    # Make predictions using the Validation set
    valid_perf = evaluate_logreg_model(model, x_valid, valid_y)
    print "Performance on Validation Set", valid_perf

    # Make predictions using test set
    test_perf = evaluate_logreg_model(model, x_test, test_y)
    print "Performance on Test Set", test_perf

    return model, {
        'train': performance_data_training,
        'valid': performance_data_valid,
        'test': performance_data_test,
    }


def optimL2_lambda(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict):
    models = []
    max_valid_perf = []
    step = 0.001
    for i in range(11):
        reg_lambda = i * step
        model, perf = part4(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict, reg_lambda)
        models.append(model)
        max_valid_perf.append(perf['valid'][-1])

    max_perf = max(enumerate(max_valid_perf), key=lambda x: x[1])[0]
    return models[max_perf], max_perf * step


def get_top_bottom_word_weights_logreg(model, word_list, stop_words=list()):
    weights = model.parameters().next().data.numpy()[0]
    weights_sorted_idx = sorted(range(len(weights)), key=weights.__getitem__)

    words_sorted = [word_list[i] for i in weights_sorted_idx if word_list[i] not in stop_words]

    words_top10_pos = words_sorted[:-11:-1]
    words_top10_neg = words_sorted[:10]

    return words_top10_pos, words_top10_neg


def format_list_as_tex(stats):
    for items in stats:
        print "\\begin{enumerate}"
        for i in items:
            print "\t\\item {}".format(i)
        print "\\end{enumerate}"


########################### Part 7 ############################
def part7(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict, max_depth):
    train_x, train_y = vectorize_headlines(real_training, fake_training, unique_words_dict)
    valid_x, valid_y = vectorize_headlines(real_validation, fake_validation, unique_words_dict)
    test_x, test_y = vectorize_headlines(real_test, fake_test, unique_words_dict)

    # Uses the Gini Impure index
    clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth=max_depth, random_state = 100, min_samples_leaf=5)
    clf_gini = clf_gini.fit(train_x, train_y)
    y_pred_train_gini = clf_gini.predict(train_x)
    score_train_gini = accuracy_score(train_y, y_pred_train_gini)

    y_pred_valid_gini = clf_gini.predict(valid_x)
    score_valid_gini = accuracy_score(valid_y, y_pred_valid_gini)

    y_pred_test_gini = clf_gini.predict(test_x)
    score_test_gini = accuracy_score(test_y, y_pred_test_gini)

    # Use the criterion information gain
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=max_depth, random_state = 100, min_samples_leaf=15)
    clf_entropy = clf_entropy.fit(train_x, train_y)
    y_pred_train_entropy = clf_entropy.predict(train_x)
    score_train_entropy = accuracy_score(train_y, y_pred_train_entropy)

    y_pred_valid_entropy = clf_entropy.predict(valid_x)
    score_valid_entropy = accuracy_score(valid_y, y_pred_valid_entropy)

    y_pred_test_entropy = clf_entropy.predict(test_x)
    score_test_entropy = accuracy_score(test_y, y_pred_test_entropy)

    return score_train_gini, score_valid_gini, score_test_gini, score_train_entropy, score_valid_entropy, score_test_entropy, clf_entropy


def plot_learning_curve_part_7(depth_values, score_train_values, score_valid_values, score_test_values, filename="Part7MaxDepthGraph"):
    train_vals = score_train_values
    validation_vals = score_valid_values
    test_vals = score_test_values

    x_axis = depth_values

    fig = plt.figure()
    plt.plot(x_axis, train_vals, 'r-', label="Training Set")
    plt.plot(x_axis, validation_vals, 'y-', label="Validation Set")
    plt.plot(x_axis, test_vals, label="Test Set")

    plt.xlabel("Depth Values")
    plt.ylabel("Accuracy of the Decision Tree")
    plt.title("Accuracy of the Decision Tree vs Depth Values")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    real_training, real_validation, real_test = load_headlines(real_filename)
    fake_training, fake_validation, fake_test = load_headlines(fake_filename)

    ################ Part 2 ##############################
    print "Finding optimal m and p value for Naive Bayes model"

    m = 2
    p = 0.2

    # Uncomment the following to tune the model
    # performance_report = tune_model(real_training, fake_training, real_validation, fake_validation)
    # m, p = max(performance_report, key=performance_report.get)
    # print "The optimal m and p value is", (m, p)

    print "Training the naive bayes model"
    model = train_model(real_training, fake_training, m, p)
    print get_performance(model, real_validation, fake_validation)

    print "Getting the top and bottom words"
    topbottom = get_top_bottom_word_weights(model)
    topbottom_stop = get_top_bottom_word_weights(model, ENGLISH_STOP_WORDS)

    print "Part 3(a):"
    format_list_as_tex(topbottom)

    print "Part 3(b):"
    format_list_as_tex(topbottom_stop)

    ########################### Part 4 ################################
    wordlist = get_wordlist(real_training, fake_training)

    unique_words_dict = {wordlist[i]: i for i in range(len(wordlist))}

    l2_lambda = 0.007

    # Uncomment the following to tune the model
    # model, l2_lambda = optimL2_lambda(real_training, fake_training, real_validation, fake_validation, real_test,
    #                                   fake_test, unique_words_dict)

    model, performance_data = part4(real_training, fake_training, real_validation, fake_validation, real_test,
                                    fake_test, unique_words_dict, l2_lambda)
    plot_learning_curve(performance_data)
    logreg_stats = get_top_bottom_word_weights_logreg(model, wordlist)
    logreg_stats_stop = get_top_bottom_word_weights_logreg(model, wordlist, ENGLISH_STOP_WORDS)

    print "Part 6(a):"
    format_list_as_tex(topbottom)

    print "Part 6(b):"
    format_list_as_tex(topbottom_stop)


    ############################## Part 7 ##############################
    depth_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    score_train_values = []
    score_valid_values = []
    score_test_values = []

    for max_depth in depth_values:
        (
            score_train_gini,
            score_valid_gini,
            score_test_gini,
            score_train_entropy,
            score_valid_entropy,
            score_test_entropy,
            clf_entropy,
        ) = part7(real_training, fake_training, real_validation, fake_validation, real_test, fake_test,
                  unique_words_dict, max_depth)
        print "##################################################################"
        print "max_depth is ", max_depth
        print "score_train_gini is ", score_train_gini
        print "score_valid_gini is ", score_valid_gini
        print "score_test_gini is ", score_valid_gini
        print "score_train_entropy is ", score_train_entropy
        print "score_valid_entropy is ", score_valid_entropy
        print "score_test_entropy is ", score_valid_entropy
        score_train_values.append(score_train_entropy)
        score_valid_values.append(score_valid_entropy)
        score_test_values.append(score_test_entropy)

    plot_learning_curve_part_7(depth_values, score_train_values, score_valid_values, score_test_values)

    (
        score_train_gini,
        score_valid_gini,
        score_test_gini,
        score_train_entropy,
        score_valid_entropy,
        score_test_entropy,
        clf_entropy,
    ) = part7(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict,
              25)

    # Uncomment the following if GraphViz is installed, and we want to regenerate the graphs
    # dot_data = tree.export_graphviz(clf_entropy, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("part7b")

    important_words = []
    important_words.append(wordlist[1332])
    important_words.append(wordlist[4421])
    important_words.append(wordlist[4269])
    important_words.append(wordlist[2008])
    important_words.append(wordlist[2974])
    important_words.append(wordlist[75])
    important_words.append(wordlist[2954])

    print "important_words is", important_words
