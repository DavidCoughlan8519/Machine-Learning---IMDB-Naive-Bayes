import os
import math
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



#unque words that appear in both the positive and negative files
vocab = set()
#stop_words = set()
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
wnl = WordNetLemmatizer()

# -------------------------------------------------------------------------------------------------
# Cleanse contents of a string array
# Returns: Cleaned array of strings
# -------------------------------------------------------------------------------------------------
def clean_words(word):
    #clean_word = re.split("\W+", word)
    #clean_word = word.translate(string.punctuation).strip()
    #clean_word = ps.stem(word)
    clean_word = wnl.lemmatize(word)
    return clean_word

# -------------------------------------------------------------------------------------------------
# Removes any stop words from the vocab set
# also populates a set of stop words for use with reading the tests
# Returns: Cleaned set without stop words
# -------------------------------------------------------------------------------------------------
def remove_stop_words():
    f = open("stopwords.txt",'r',encoding="utf8")
    file = f.readlines()

    for word in file:
        stop_words.add(word)
        if word is not None and word in vocab:
             vocab.remove(word)
    return vocab

#-------------------------------------------------------------------------------------------------
#Loads in the files from the path given and populates the set of unique words from reading the files
#-------------------------------------------------------------------------------------------------
def load_files_for_training(path):
    print("Loading training files...")
    listing = os.listdir(path)
    for file in listing:
        file = open(path + file,'r',encoding = "utf8")
        words = file.read().lower().split()

        for word in words:
            if word is not None:
                vocab.add(clean_words(word))
    print("Finished loading training files...")


#-------------------------------------------------------------------------------------------------
#Calculates the frequency of the word from the number of occurrences of its appearance in the files
#returns: a dictionary with the word as the key amd the frequency as the value
#-------------------------------------------------------------------------------------------------
def calculate_frequency(vocab,path):
    frequencies = dict.fromkeys(vocab,0)
    words = []
    listing = os.listdir(path)
    for file in listing:
            file = open(path + file,'r',encoding = "utf8")
            words = file.read().lower().split()
            for word in words:
                if word in frequencies:
                    frequencies[word] = frequencies.get(word) + 1
    return frequencies

#-------------------------------------------------------------------------------------------------
#Calculates the probability of the word from the occurrence of its appearance
#returns: a dictionary with the word as the key amd the probability as the value
#-------------------------------------------------------------------------------------------------
def calculate_probability(frequency, vocab):
    prob_dict = dict.fromkeys(vocab, 0)
    total_frequencies = sum(frequency.values())

    for word in vocab:
        if word in frequency:
            prob_dict[word] = (frequency.get(word) + 1) / (total_frequencies + len(vocab))
    return prob_dict

#-------------------------------------------------------------------------------------------------
#Calculates the naive bayes classification of the whole review file from adding the probabilities
#of each word that appears in the review based on the dictionary passed in (positive or negative)
#returns: sum of all the probabilities
#-------------------------------------------------------------------------------------------------
def naive_bayes_classification(probability, review,sum_total_files,total_pos_file_num,total_neg_file_num,test):
    sum_of_probabilities = 0
    pc = 0
    if test =='pos':
        pc = (total_pos_file_num/sum_total_files)
    else:
        pc = (total_neg_file_num / sum_total_files)
    for word in review:
        temp = clean_words(word)
        if temp in probability and temp not in stop_words:
            sum_of_probabilities += math.log(probability[temp],10)
    return math.log(pc,10) + sum_of_probabilities


#-------------------------------------------------------------------------------------------------
#Counts the number of files in a directory for use in the formula
#returns: sum of all files in the directory
#-------------------------------------------------------------------------------------------------
def count_num_files_in_dir(path):
    total = 0
    listing = os.listdir(path)
    for file in listing:
        total+=1
    return total
#-------------------------------------------------------------------------------------------------
#Loads in the files from the test reviews path given.
#-------------------------------------------------------------------------------------------------
def load_files_for_testing(test,positive_probability,negative_probability):
    print("Starting loading for testing...")
    test_path = "smallTest\\" + test + "\\"
    sum_predicted_positive = 0
    sum_predicted_negative = 0
    pos_review_prob = 0
    neg_review_prob = 0
    total_pos_file_num = 0
    total_neg_file_num = 0

    if test == 'neg':
        test_path = "smallTest\\neg\\"

    total_pos_file_num = count_num_files_in_dir("smallTest\\pos\\")
    total_neg_file_num = count_num_files_in_dir("smallTest\\neg\\")
    sum_total_files = total_pos_file_num + total_neg_file_num
    listing = os.listdir(test_path)
    for file in listing:
        file_data = open(test_path + file, 'r', encoding="utf8")
        review  = file_data.read().lower().split()

        pos_review_prob = naive_bayes_classification(positive_probability,review,sum_total_files,total_pos_file_num,total_neg_file_num,'pos')
        neg_review_prob = naive_bayes_classification(negative_probability,review,sum_total_files,total_pos_file_num,total_neg_file_num,'neg')

        if pos_review_prob > neg_review_prob:
            sum_predicted_positive += 1
        else:
            sum_predicted_negative += 1

    if test == "pos":
        print("Positive predictions: " + str((sum_predicted_positive / 1000) * 100) + "%")
    else:
        print("Negative predictions: " + str((sum_predicted_negative / 1000) * 100) + "%")


def main():
    #Load files for training and populate the set of the vocab
    load_files_for_training('LargeIMDB\\pos\\')
    load_files_for_training('LargeIMDB\\neg\\')

    #remove stop words from the vocab set
    remove_stop_words()

    #get the frequency dictionaries
    positive_frequency = calculate_frequency(vocab,'LargeIMDB\\pos\\')
    negative_frequency = calculate_frequency(vocab,'LargeIMDB\\neg\\')

    #get the probability dictionaries
    positive_probability = calculate_probability(positive_frequency,vocab)
    negative_probability = calculate_probability(negative_frequency,vocab)

    #test the data model on unseen reviews
    #known positive reviews
    load_files_for_testing('pos',positive_probability,negative_probability)
    #known negative reviews
    load_files_for_testing('neg',positive_probability,negative_probability)

main()