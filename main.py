import mnist
import numpy as np
import pandas as pd

import urllib
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
from scipy.misc import face

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score

from mnist import MNIST

mndata = MNIST('samples')

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
#print len(images)
#print len(labels)
#print labels[0:101]
#index =0
#print(mndata.display(train_images[index]))
print(train_images[2])
print len(train_images[2])
type(train_images[2])
img_reshaped = np.reshape(train_images[100],(28,28))
imgplot = plt.imshow(img_reshaped)

plt.show()
## bernoulli NB which is good if we convert data into binary

BernNB = BernoulliNB(binarize= True)
BernNB.fit(train_images,train_labels)
print (BernNB)

#print(test_labels)
test_labels_list = test_labels.tolist()
#print(test_labels_list )
test_expect = test_labels_list
y_pred = BernNB.predict(test_images)
#print y_pred
print accuracy_score(y_pred,test_expect)

### multinomial naive bayes classifier, i believe here we are counting freq
# of word occur so multinomial should be the best

MultiNB = MultinomialNB()
MultiNB.fit(train_images,train_labels)
print (MultiNB)

y_pred = MultiNB.predict(test_images)
#print y_pred
#print test_expect
print accuracy_score(y_pred,test_expect)

### gaussian NB, if the 48 predictor are normally distributed (in column)
# then gaussian can also give good results

GausNB = GaussianNB()
GausNB.fit(train_images,train_labels)
print (GausNB)

y_pred = GausNB.predict(test_images)
#print y_pred
#print test_expect
print accuracy_score(y_pred,test_expect) # gaussian does not give good results

## bernoulli NB which is good if we convert data into binary, now playing with
# parameters give good results

BernNB = BernoulliNB(binarize= 0.1)
BernNB.fit(train_images,train_labels)
print (BernNB)

y_pred = BernNB.predict(test_images)
#print y_pred
print accuracy_score(y_pred,test_expect)