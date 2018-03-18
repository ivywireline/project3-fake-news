import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import cPickle as pickle

from collections import defaultdict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

real_filename = 'clean_real.txt'
fake_filename = 'clean_fake.txt'

# Key is a unique word string. Value is the index of the word in unique_words_set
# unique_words_dict = {}


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
        # if word in ENGLISH_STOP_WORDS: continue
        if word in headline_split:
            logprob_real += math.log(probabilities_real[word])
            logprob_fake += math.log(probabilities_fake[word])
        else:
            logprob_real += math.log(1 - probabilities_real[word])
            logprob_fake += math.log(1 - probabilities_fake[word])
    real_prob = math.exp(logprob_real) * real_class_prob
    fake_prob = math.exp(logprob_fake) * fake_class_prob
    # print real_prob, fake_prob
    return real_prob > fake_prob

def tune_model(real_training, fake_training, real_validation, fake_validation):
    performance_report = {}
    m = 1
    while m <= 20:
        p = 0.1
        while p <= 1:
            model = train_model(real_training, fake_training, m, p)
            performance = get_performance(model, real_validation, fake_validation)
            print m, p, performance
            performance_report[(m, p)] = performance
            p += 0.1
        m += 1

    print "The m and p value is", max(performance_report, key=performance_report.get)

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

def get_top_bottom_word_occurrences(model, stop_words=list()):
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
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)
    idx = 0
    unique_words_dict = {}

    for word in unique_words_set:
        unique_words_dict[word] = idx
        idx += 1

    return unique_words_dict


def get_train(real_training, fake_training, unique_words_dict):
    # 1 means real. 0 means fake.
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)

    # Number of features is the number of the unique words in the training set
    batch_xs = np.zeros((0, unique_words_number))
    # There are only 2 classes - real or fake
    batch_y_s = []

    for headline in real_training:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            index = unique_words_dict[word]
            vector_headline[index] = 1
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(1)

    for headline in fake_training:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            index = unique_words_dict[word]
            vector_headline[index] = 1
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(0)

    return batch_xs, batch_y_s


def get_validation(real_training, fake_training, real_validation, fake_validation, unique_words_dict):
    # 1 means real. 0 means fake.
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)

    # Number of features is the number of the unique words in the validation set
    batch_xs = np.zeros((0, unique_words_number))
    # There are only 2 classes - real or fake
    batch_y_s = []

    for headline in real_validation:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 1
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(1)

    for headline in fake_validation:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 0
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(0)

    return batch_xs, batch_y_s


def get_test(real_training, fake_training, real_test, fake_test, unique_words_dict):
    # 1 means real. 0 means fake.
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)

    # Number of features is the number of the unique words in the validation set
    batch_xs = np.zeros((0, unique_words_number))
    # There are only 2 classes - real or fake
    batch_y_s = []

    for headline in real_test:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 1
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(1)


    for headline in fake_test:
        # Vector simulating the headline
        vector_headline = np.zeros(unique_words_number)
        headline_words = headline.split(" ")
        for word in headline_words:
            if word in unique_words_dict:
                index = unique_words_dict[word]
                vector_headline[index] = 0
        batch_xs = np.vstack((batch_xs, vector_headline))
        batch_y_s.append(0)

    return batch_xs, batch_y_s


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


def part4(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict):
    # 1 means real. 0 means fake.
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)

    train_x, train_y = get_train(real_training, fake_training, unique_words_dict)
    train_y = np.array(train_y)
    valid_x, valid_y = get_validation(real_training, fake_training, real_test, fake_test, unique_words_dict)
    valid_y = np.array(valid_y)
    test_x, test_y = get_test(real_training, fake_training, real_test, fake_test, unique_words_dict)
    test_y = np.array(test_y)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_x_var = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    valid_x_var = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
    test_x_var = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    # Hyper Parameters
    input_size = unique_words_number
    num_classes = 1
    num_epochs = 3
    batch_size = 32
    learning_rate = 0.0001
    iter_limit = 3000

    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    batches = np.random.permutation(range(train_x.shape[0]))

    intermediate_perf = {
        'train': [],
        'valid': [],
        'test': [],
    }

    for epoch in range(num_epochs):
        idx = 0

        headline_tensor = torch.from_numpy(train_x).double()
        headline_var = Variable(headline_tensor, requires_grad=False).type(dtype_float)
        # labels_var = Variable(labels)
        label_tensor = torch.from_numpy(train_y).double()
        label_var = Variable(label_tensor, requires_grad=False).type(dtype_float)

        # Forward + Backward + Optimize
        # optimizer.zero_grad()
        output = model(headline_var)

        loss = loss_fn(output, label_var)
        loss.backward()
        optimizer.step()

        print ('Epoch: [%d/%d], Loss: %.4f'
               % (epoch+1, num_epochs, loss.data[0]))

    # Make predictions using set
    x = Variable(torch.from_numpy(test_x).double(), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy().argmax(axis=1)

    print "y_pred is ", y_pred

    print "test_y is ", test_y

    print np.mean(y_pred == test_y)

    return model


if __name__ == '__main__':
    random.seed(0)
    real_training, real_validation, real_test = load_headlines(real_filename)
    fake_training, fake_validation, fake_test = load_headlines(fake_filename)

    # # These parameters need lots of tweaking
    # m = 1.0
    # p = 1.0
    #
    # model = train_model(real_training, fake_training, m, p)
    #
    # # performance_training, performance_test, performance_validation_real, performance_validation_fake, performance_validation_total = get_total_performance(model, real_training, fake_training, real_test, fake_test, real_validation, fake_validation)
    #
    # # process_headlines(real_training, fake_training)
    # # batch_xs, batch_y_s = get_train(real_training, fake_training)
    #
    # # print "batch_xs", batch_xs
    # # print "batch_y_s", batch_y_s
    #
    m = 2
    p = 0.2

    model = train_model(real_training, fake_training, m, p)
    print get_performance(model, real_validation, fake_validation)
    #
    #
    #
    # # print tune_model(real_training, fake_training, real_validation, fake_validation)
    #
    topbottom = get_top_bottom_word_occurrences(model)
    topbottom_stop = get_top_bottom_word_occurrences(model, ENGLISH_STOP_WORDS)

    for items in topbottom:
        print "\\begin{enumerate}"
        for i in items:
            print "\t\\item {}".format(i)
        print "\\end{enumerate}"

    for items in topbottom_stop:
        print "\\begin{enumerate}"
        for i in items:
            print "\t\\item {}".format(i)
        print "\\end{enumerate}"

    # high_fake = [a for a in fake_counts if a in real_counts and fake_counts[a] > 3 and fake_counts[a] > real_counts[a]]

    # print high_fake

    # with open('real_word_counts.json', 'w') as real_word_counts:
    #     json.dump(real_counts, real_word_counts)

    # with open('fake_word_counts.json', 'w') as fake_word_counts:
    #     json.dump(fake_counts, fake_word_counts)

    ############################ Part 4 ################################

    # dictionary_unique_words = process_headlines(real_training, fake_training)
    # pickle.dump(dictionary_unique_words, open("Part4Dictionary.pkl", 'wb'))
    # unique_words_dict = pickle.load(open("Part4Dictionary.pkl"))
    #
    # model = part4(real_training, fake_training, real_validation, fake_validation, real_test, fake_test, unique_words_dict)
