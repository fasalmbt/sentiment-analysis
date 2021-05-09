from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

train_df = pd.read_csv('./dataset/train.csv')
test_df = pd.read_csv('./dataset/test.csv')

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

train_vec = vectorizer.fit_transform(train_df['Content'])
test_vec = vectorizer.transform(test_df['Content'])

classifier = LinearSVC()
t1 =time.time()
classifier.fit(train_vec, train_df['Label'])
t2 = time.time()
prediction = classifier.predict(test_vec)
t3 = time.time()
time_to_train = t2 - t1
time_to_predict = t3 - t2

print("-------------------------------Results-------------------------------\n")
print(f"Training time: {time_to_train}")
print(f"Prediction time: {time_to_predict}")
report = classification_report(test_df['Label'], prediction, output_dict=True)
print('Positive: ', report['positive'])
print('Negative: ', report['negative'])
print("\n---------------------------------------------------------------------\n")

pickle.dump(vectorizer, open('./models/vectorizer.sav', 'wb'))

pickle.dump(classifier, open('./models/classifier.sav', 'wb'))

print("Dumping completed!")
