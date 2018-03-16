import math
import random
import json

from collections import defaultdict

real_filename = 'clean_real.txt'
fake_filename = 'clean_fake.txt'

def get_wordlist(*args):
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
        already_found = []
        for word in line.split(' '):
            if word not in already_found:
                word_counts[word] += 1
                already_found.append(word)

    return word_counts

def train_model(real_headlines, fake_headlines, m, p):
    word_list = get_wordlist(real_headlines, fake_headlines)
    real_counts = count_word_occurrance(real_training)
    fake_counts = count_word_occurrance(fake_training)
    probabilities_real = {}
    probabilities_fake = {}
    for word in word_list:
        if word in real_counts:
            probabilities_real[word] = (real_counts[word] + m * p) / float(len(real_headlines) + m)
        if word in fake_counts:
            probabilities_fake[word] = (fake_counts[word] + m * p) / float(len(fake_headlines) + m)

    return probabilities_real, probabilities_fake, m, p, len(real_headlines), len(fake_headlines)

def predict_model(model, headline):
    probabilities_real, probabilities_fake, m, p, real_count, fake_count = model
    probability_real = 0.0
    probability_fake = 0.0
    for word in headline.split(' '):
        probability_real += math.log((probabilities_real.get(word, 0) + m * p) / float(real_count + m))
        probability_fake += math.log((probabilities_fake.get(word, 0) + m * p) / float(fake_count + m))
    print probability_real
    print probability_fake
    return math.exp(probability_real) > math.exp(probability_fake)


if __name__ == '__main__':
    random.seed(0)
    real_training, real_validation, real_test = load_headlines(real_filename)
    fake_training, fake_validation, fake_test = load_headlines(fake_filename)

    print 'Num training', 'Num validation', 'Num test'
    print len(real_training), len(real_validation), len(real_test)
    print len(fake_training), len(fake_validation), len(fake_test)

    # real_counts = count_words(real_training)
    # fake_counts = count_words(fake_training)

    #print get_wordlist(real_training, fake_training)

    # These parameters need lots of tweaking
    m = 100
    p = 10

    model = train_model(real_training, fake_training, m, p)
    print predict_model(model, real_validation[0])

    # high_fake = [a for a in fake_counts if a in real_counts and fake_counts[a] > 3 and fake_counts[a] > real_counts[a]]

    # print high_fake

    # with open('real_word_counts.json', 'w') as real_word_counts:
    #     json.dump(real_counts, real_word_counts)
    
    # with open('fake_word_counts.json', 'w') as fake_word_counts:
    #     json.dump(fake_counts, fake_word_counts)
