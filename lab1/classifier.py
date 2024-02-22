import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import random


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    p = util.get_word_freq(file_lists_by_category[0])
    q = util.get_word_freq(file_lists_by_category[1])

    all_words = set(p.keys()).union(set(q.keys()))
    d = len(all_words)
    num_spam = sum(p.values())
    num_ham = sum(q.values())

    for word in all_words:
        if word in p:
            p[word] = (p[word] + 1) / (num_spam + d)
        else:
            p[word] = 1 / (num_spam + d)

        if word in q:
            q[word] = (q[word] + 1) / (num_ham + d)
        else:
            q[word] = 1 / (num_ham + d)

    return p, q


def classify_new_email(filename, probabilities_by_category, prior_by_category, threshold=1):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    word_dict = util.get_word_freq([filename])
    spam_sum = 0
    ham_sum = 0
    p = probabilities_by_category[0]
    q = probabilities_by_category[1]
    for k, v in word_dict.items():
        if k in p:
            spam_sum += np.log(p[k]) * v
        if k in q:
            ham_sum += np.log(q[k]) * v

    spam_sum += np.log(prior_by_category[0])
    ham_sum += np.log(prior_by_category[1])

    if spam_sum > ham_sum + np.log(threshold):
        result = "spam"
    else:
        result = "ham"

    return result, (spam_sum, ham_sum)


def select_files(directory, fraction=0.75):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    random.shuffle(all_files)
    num_files = int(len(all_files) * fraction)
    return all_files[:num_files]


def plot_tradeoff(test_folder, probabilities_by_category, priors_by_category):
    thresholds = [1e-100, 1e-10, 1e-5, 0.1, 1, 50, 100, 1e4, 1e10, 1e100]

    type1_errors = []
    type2_errors = []

    for threshold in thresholds:
        performance_measures = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            label, _ = classify_new_email(filename, probabilities_by_category, priors_by_category, threshold)

            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        type1_error = performance_measures[1, 0]  # True label: Ham, Predicted: Spam
        type2_error = performance_measures[0, 1]  # True label: Spam, Predicted: Ham
        type1_errors.append(type1_error)
        type2_errors.append(type2_error)

    # print(type1_errors)
    # print(type2_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(type1_errors, type2_errors, marker='o')
    plt.title('Trade-off between Type 1 and Type 2 Errors')
    plt.xlabel('Number of Type 1 Errors')
    plt.ylabel('Number of Type 2 Errors')
    plt.grid(True)
    plt.savefig('nbc.pdf', format='pdf')
    # plt.show()


if __name__ == '__main__':

    ############################CHANGE YOUR STUDENT ID###############################
    # student_number = 12345678  # Replace with the actual student number
    student_number = 1006212821
    random.seed(student_number)
    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    if student_number % 2 == 0:
        test_folder = "data/testing2"
    else:
        test_folder = "data/testing1"

    # generate the file lists for training
    file_lists = []
    file_lists = [select_files(folder) for folder in (spam_folder, ham_folder)]

    # Learn the distributions
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2, 2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label, log_posterior = classify_new_email(filename, probabilities_by_category, priors_by_category)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0], totals[0], correct[1], totals[1]))
    # plot_tradeoff(test_folder, probabilities_by_category, priors_by_category, smooth_vals)
