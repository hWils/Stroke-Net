"""
Takes in two 2D arrays, one for training and one for test, performs a series of traditional machine learning classifications; knn, lr, svm, random forest and lda.

"""

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
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
