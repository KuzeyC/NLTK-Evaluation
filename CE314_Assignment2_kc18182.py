# Kuzey Cimen - kc18182 - 1803189
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.metrics import precision, recall, f_measure
import random
from collections import defaultdict


# 1: Text classification
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)



# 2: Pre-processing
# Remove stop words and lowercase words
stop_words = set(stopwords.words('english'))
filtered = " ".join([w.lower() for w in movie_reviews.words() if w.lower() not in stop_words])

# Remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
no_punc = tokenizer.tokenize(filtered)

# Lemmatization
lem = WordNetLemmatizer()
l = [lem.lemmatize(w) for w in no_punc]



# 3: Feature selection
# Feature finding function
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


all_words = FreqDist(l)
word_features = list(all_words.keys())[:3000]
featuresets = [(find_features(rev), category) for (rev, category) in documents]



# 4: Evaluation
# Evaluation function.
def evaluate(type):
    temp_train = defaultdict(set)
    temp_test = defaultdict(set)
    for i, (feats, label) in enumerate(testing_set):
        temp_train[label].add(i)
        temp_test[classifier.classify(feats)].add(i)
    return (temp_train[type], temp_test[type])


# Naive Bayes - 90% training set, 10% testing set
# Training Set to train the classifier.
training_set = featuresets[:1800]

# Testing Set to test the classifier.
testing_set = featuresets[1800:]

# Classifier
classifier = NaiveBayesClassifier.train(training_set)

# Accuracy
print("Classifier accuracy percent:",
      (accuracy(classifier, testing_set))*100)

# Pos/Neg Evaluations
pos_evaluation = evaluate("pos")
neg_evaluation = evaluate("neg")

print()

# Evaluation for positive.
print('pos precision:', precision(pos_evaluation[0], pos_evaluation[1]))
print('pos recall:', recall(pos_evaluation[0], pos_evaluation[1]))
print('pos F-measure:', f_measure(pos_evaluation[0], pos_evaluation[1]))

print()

# Evaluation for negative.
print('neg precision:', precision(neg_evaluation[0], neg_evaluation[1]))
print('neg recall:', recall(neg_evaluation[0], neg_evaluation[1]))
print('neg F-measure:', f_measure(neg_evaluation[0], neg_evaluation[1]))
