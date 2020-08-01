from google.colab import files
uploaded = files.upload()

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import scipy.sparse as sparse

import io

df=pd.read_csv('news.csv')
df.shape
df.head()
df

labels=df.label
labels.head()

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
X=df['text']
print(X.shape)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_vectorizer.fit(X)
tfidf_data=tfidf_vectorizer.transform(X)
tfidf_train=tfidf_vectorizer.transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

tfidf_data.shape


feature_names=tfidf_vectorizer.get_feature_names
feature_names.shape

from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=2500, random_state=42)
svd.fit(tfidf_data)
x_train_svd=svd.transform(tfidf_train)
x_test_svd=svd.transform(tfidf_test)

plt.plot(np.cumsum(svd.explained_variance_ratio_))

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
print(y_pred)

pac.fit(x_train_svd,y_train)
y_pred_svd=pac.predict(x_test_svd)
print("The Classification Report after using Truncated SVD is:")
print(classification_report(y_test,y_pred_svd,target_names=['FAKE','REAL']))
print(svd.explained_variance_ratio_.sum())
print("Confusion Matrix after using Truncated SVD is:")
print(confusion_matrix(y_test,y_pred_svd,labels=['FAKE','REAL']))

confusion_matrix(y_test ,y_pred, labels=['FAKE','REAL'])

print(classification_report(y_test,y_pred,labels=['FAKE' , 'REAL']))

import matplotlib.pyplot as plt
labels='True Positives','True Negative','False Positives','False Negatives'
sizes=[591,585,44,47]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0.1,0,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#USING SVM for classification
from sklearn import svm
svm_model=svm.SVC(kernel='linear')
svm_model.fit(x_train_svd,y_train)

y_pred_svm=svm_model.predict(x_test_svd)

print(classification_report(y_test,y_pred_svd,target_names=target_names))

print(confusion_matrix(y_test,y_pred_svd,labels=['FAKE','REAL']))

labels='True Positives','True Negative','False Positives','False Negatives'
sizes=[590,580,48,49]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0.1,0,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

plt.plot(np.cumsum(svd.explained_variance_ratio_))

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

log_model=LogisticRegression()
log_model.fit(x_train_svd,y_train)
ylog_predict=log_model.predict(x_test_svd)
print("Confusion Matrix:")
print(confusion_matrix(y_test,ylog_predict,labels=['FAKE','REAL']))
print("Classification Report ")
print(classification_report(y_test,ylog_predict,labels=['FAKE','REAL']))

gaussian_nb= GaussianNB()
gaussian_nb.fit(x_train_svd,y_train)
y_predict =gaussian_nb.predict(x_test_svd)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_predict,labels=['FAKE','REAL']))
print("Classification Report:")
print(classification_report(y_test , y_predict , labels=['FAKE','REAL'] ))

transform_data=svd.transform(tfidf_data)
labels.shape

from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
color_choice = sns.color_palette("bright", 2)
tsne = TSNE(n_components=2)
transform_data=StandardScaler().fit_transform(transform_data)

X_tSNE = tsne.fit_transform(transform_data)
sns.scatterplot(X_tSNE[:,0], X_tSNE[:,1], hue=labels, legend='full', palette=color_choice)