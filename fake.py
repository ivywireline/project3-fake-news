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

from collections import defaultdict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

real_filename = 'clean_real.txt'
fake_filename = 'clean_fake.txt'

# Key is a unique word string. Value is the index of the word in unique_words_set
unique_words_dict = {}


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
            if probabilities_real[word] > 1:
                raise ValueError
        if word in fake_counts:
            probabilities_fake[word] = (fake_counts[word] + m * p) / float(len(fake_headlines) + m)
            if probabilities_fake[word] > 1:
                raise ValueError

    return probabilities_real, probabilities_fake, m, p, len(real_headlines), len(fake_headlines), word_list

def predict_model(model, headline):
    probabilities_real, probabilities_fake, m, p, real_count, fake_count, word_list = model
    logprob_real = 0.0
    logprob_fake = 0.0
    real_class_prob = float(real_count) / (real_count + fake_count)
    fake_class_prob = float(fake_count) / (real_count + fake_count)
    headline_split = headline.split(' ')
    for word in word_list:
        if word in headline_split:
            if word in probabilities_real:
                logprob_real += math.log(probabilities_real[word])
            if word in probabilities_fake:
                logprob_fake += math.log(probabilities_fake[word])
        else:
            if word in probabilities_real:
                logprob_real += math.log(1 - probabilities_real[word])
            if word in probabilities_fake:
                logprob_fake += math.log(1 - probabilities_fake[word])
        # if word in ENGLISH_STOP_WORDS: continue
    real_prob = math.exp(logprob_real) * real_class_prob
    fake_prob = math.exp(logprob_fake) * fake_class_prob
    # print real_prob, fake_prob
    return real_prob > fake_prob

def tune_model(real_training, fake_training, real_validation, fake_validation):
    performance_report = {}
    m = 1
    while m <= 10:
        p = 0.0
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

############################# Logistic Regression #############################
def process_headlines(real_training, fake_training):
    # Get the unique words list. Its index is important for feature vector representation.
    unique_words_set = get_wordlist(real_training, fake_training)
    unique_words_number = len(unique_words_set)
    idx = 0

    for word in unique_words_set:
        unique_words_dict[word] = idx
        idx += 1


def get_train(real_training, fake_training):
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

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

def part4():
    # Hyper Parameters
    input_size = 784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    random.seed(0)
    real_training, real_validation, real_test = load_headlines(real_filename)
    fake_training, fake_validation, fake_test = load_headlines(fake_filename)

    # These parameters need lots of tweaking
    m = 1.0
    p = 1.0

    model = train_model(real_training, fake_training, m, p)

    performance_training, performance_test, performance_validation_real, performance_validation_fake, performance_validation_total = get_total_performance(model, real_training, fake_training, real_test, fake_test, real_validation, fake_validation)

    process_headlines(real_training, fake_training)
    batch_xs, batch_y_s = get_train(real_training, fake_training)

    print "batch_xs", batch_xs
    print "batch_y_s", batch_y_s

    m = 1.0
    p = 1.0

    model = train_model(real_training, fake_training, m, p)

    print tune_model(real_training, fake_training, real_validation, fake_validation)

    # high_fake = [a for a in fake_counts if a in real_counts and fake_counts[a] > 3 and fake_counts[a] > real_counts[a]]

    # print high_fake

    # with open('real_word_counts.json', 'w') as real_word_counts:
    #     json.dump(real_counts, real_word_counts)

    # with open('fake_word_counts.json', 'w') as fake_word_counts:
    #     json.dump(fake_counts, fake_word_counts)
