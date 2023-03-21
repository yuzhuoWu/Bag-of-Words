import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import nltk.data
from transformers import BertTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.models import Sequential


num_features = 300
max_features = 6000
embed_dim = 128

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 1000. == 0.:
            print("Review %d of %d ", counter, len(reviews))

        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)

        counter = counter + 1.
    return reviewFeatureVecs

def review_to_wordlist(review, remove_stopwords=False):
     # 删除 HTML标记
     review_text = BeautifulSoup(review).get_text()

     # 删除 non-letters
     review_text = re.sub("[^a-zA-Z]", " ", review_text)

     # 转换为小写,并拆分为单独的单词
     words = review_text.lower().split()

     # 有选择地删除 stop words
     if remove_stopwords:
          stops = set(stopwords.words("english"))
          words = [w for w in words if not w in stops]

     return words

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, limiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    unlabeled = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,delimiter="\t", quoting=3)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences = []

    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    for review in unlabeled["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    train['clean_review'] = train['review'].apply(lambda x: getCleanReviews(x))
    test['clean_review'] = test['review'].apply(lambda x: getCleanReviews(x))
    train_text = train.clean_review.values
    test_text = test.clean_review.values

    length = []
    all_words = ' '.join(train_text)
    dist = nltk.FreqDist(all_words)
    unique_word = len(dist)

    for i in train_text:
        words = word_tokenize(i)
        l = len(words)
        length.append(l)
    max_review = np.max(length)
    max_words = max_review

    x_train = tokenizer.texts_to_sequences(train_text)
    x_train = pad_sequences(x_train, maxlen=max_words)
    y_train = train['sentiment']

    model = Sequential()
    model.add(Embedding(max_features, embed_dim))
    model.add(LSTM(64, dropout=0.04, recurrent_dropout=0.4, return_sequences=True))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=300, epochs=10, validation_split=0.2)

    # 获取均值向量
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    # 随机森林分类测试集
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, y_train)

    result = forest.predict(testDataVecs)

    # 输出测试结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("LSTM.csv", index=False, quoting=3)
