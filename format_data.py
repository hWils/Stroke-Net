# -*- coding: utf-8 -*-
"""
This takes the data for one ppt in one session and organises it into
two arrays, one for left and one for right motor imagery data. 
All data not related to the MI trials is disregarded.
"""
import numpy as np
import scipy.io as spio



"""
path_training = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/stroke/data/P1_pre_training.mat'
path_test = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/stroke/data/P1_pre_test.mat'
train_mat = spio.loadmat(path_training, squeeze_me=True)
test_mat = spio.loadmat(path_test, squeeze_me=True)



def describe_organise_data(mat):
    print("The keys of this data dictionary for this ppt are :", mat.keys())
    triggers = mat['trig']
    fs = mat['fs']
    eeg_data = mat['y']
    print("There are ", len(triggers), " triggers") 
    print("sampling frequency is ", mat['fs'])
    print("Length of data set is ", len(eeg_data))
    return triggers, fs, eeg_data


# get indices that indicate change in blocks, such as 000s then 1111s in order to split these
def format_data(triggers, fs, eeg_data):
    v=np.asarray(triggers)
    indices = np.where(np.diff(v,prepend=np.nan))[0] 

    listvalues = []
    listclasses = []
    ind = 0
    
    for ind in range(0, len(indices)-1):
        listclasses.append(triggers[indices[ind]]) # means we know which class each block relates to
        listvalues.append(eeg_data[indices[ind]:indices[ind+1]]) # gets the block as a subset
         
    left_class = []
    left_value = []
    right_class = []
    right_value = []
    for i,x in enumerate(listclasses):
        if x == 1:
            left_class.append(listclasses[i])
            left_value.append(listvalues[i])
        elif x == -1:
            right_class.append(listclasses[i])
            right_value.append(listvalues[i])
        
 
    left_data = np.array([np.array(x) for x in left_value])
    right_data = np.array([np.array(x) for x in right_value])
    left_labels = np.ones(left_data.shape[0])
    right_labels = np.zeros(right_data.shape[0]) - 1
    
    data = np.concatenate((left_data,right_data))
    labels = np.concatenate((left_labels,right_labels))
    return data,labels




# splits the data into the two separate conditions left and right, removes 0 triggers
# this also means each trial should now be equal length so can be put into an array
test_triggers, test_fs, test_eeg = describe_organise_data(test_mat)
train_triggers, train_fs, train_eeg = describe_organise_data(train_mat)

train_data, train_labels =format_data(train_triggers, train_fs, train_eeg)
test_data, test_labels = format_data(test_triggers, test_fs, test_eeg )




   
    # have to flatten into two dimensions, cannot take 16 channels, 
train_data_2D = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
test_data_2D = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])



from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
svc = LDA()
csp = CSP(n_components=4, reg=None, log=True)

from sklearn.pipeline import Pipeline  # noqa

lda_clf = Pipeline([('CSP', csp), ('SVC', svc)])
lda_clf.fit(train_data, train_labels)
print("score with LDA is ", lda_clf.score(test_data, test_labels))


"""





path_training = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/P2_pre_training_caract.mat'
path_test = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/P2_pre_test_caract.mat'
train_mat = spio.loadmat(path_training, squeeze_me=True)
test_mat = spio.loadmat(path_test, squeeze_me=True)



print(train_mat.keys())
train_data = train_mat['mcaract']
test_data= test_mat['mcaract']
train_labels = train_mat['vlabel']
test_labels = test_mat['vlabel']

#print(test_labels[0:1000])


def get_unique_numbers(numbers):
    unique = []

    for number in numbers:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


print(get_unique_numbers(test_labels))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# different classifiers
# KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data, train_labels)


print("score with k's nearest neighbour is ", neigh.score(test_data, test_labels))
"""
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print(‘cv_scores mean:{}’.format(np.mean(cv_scores)))
"""

###LDA

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(train_data, train_labels)
print("score with LDA is ", lda_clf.score(test_data, test_labels))



# SVM
svm_clf = svm.SVC()
svm_clf.fit(train_data, train_labels)
print("score with svm is ", svm_clf.score(test_data, test_labels))

# LR

lr = LogisticRegression(penalty ='l2',random_state=6, max_iter=1000)
lr.fit(train_data, train_labels)
print("score with logistic regression is ", lr.score(test_data, test_labels))


## regression models, random forest

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(train_data, train_labels)
y_pred = regressor.predict(test_data)
print("score with random forest is ", regressor.score(test_data, test_labels))






"""

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))



print(train_data_2D.shape)

"""

# LSTM
"""
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score


print("Now carrying out LSTM")

model = Sequential()
model.add(LSTM(256, input_shape=(train_data.shape[1], 16)))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.001)
#chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=200, batch_size= 32,  validation_data=(test_data,test_labels))
#print(model.parameters)

test_preds = model.predict(test_data)
accuracy_score = accuracy_score(test_labels, test_preds)
print(accuracy_score)
 """


