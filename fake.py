import random

real_filename = 'clean_real.txt'
fake_filename = 'clean_fake.txt'

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

if __name__ == '__main__':
    random.seed(0)
    real_training, real_validation, real_test = load_headlines(real_filename)
    fake_training, fake_validation, fake_test = load_headlines(fake_filename)

    print 'Num training', 'Num validation', 'Num test'
    print len(real_training), len(real_validation), len(real_test)
    print len(fake_training), len(fake_validation), len(fake_test)
