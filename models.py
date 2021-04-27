import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
import datetime as dt

import keras as ks
import tensorflow as tf

import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.neighbors
import sklearn.ensemble
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_decision_regions

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import xgboost as xgb

### HELPER FUNCTIONS 

def get_popular_words(corpus, top_n):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_array = vectorizer.get_feature_names()
    top_words = sorted(list(zip(vectorizer.get_feature_names(), X.sum(0).getA1())), key=lambda x: x[1], reverse=True)[:top_n]
    result = [x[0] for x in top_words]
    return result


def get_pw_from_file(filename, column, n_top):
    df = pd.read_csv(filename)
    df['all_text'] = df['headline'] + " " + df['abstract'] + " " + df['keywords']
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) 
    df[column] = df[column].astype(str)
    df = lemmatize_column(df, column, lemmatizer, stop_words)
    popular_words = get_popular_words(df[column], n_top)
    
    return popular_words


def process_sentence(sentence, lemmatizer, stop_words):
    sentence = sentence.lower()
    tokens = list(set(word_tokenize(sentence)))   
    words = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(words)


def ord_encode(df, ordinal_features):
    # Ordinal encode all of these features
    ordinal = sklearn.preprocessing.OrdinalEncoder()
    df[ordinal_features] = ordinal.fit_transform(df[ordinal_features])
    return df


def encode_language_column(df, col_name, popular_words = []):
    vectorizer = CountVectorizer()
    nc = vectorizer.fit_transform(df[col_name])
    encoded_col = pd.DataFrame(nc.A, columns=vectorizer.get_feature_names())[popular_words]
    df = pd.concat([df.reset_index(drop=True), encoded_col.reset_index(drop=True)], axis=1)
    return df


def lemmatize_column(df, col_name, lemmatizer, stop_words):
    df[col_name] = df[col_name].map(lambda x: process_sentence(x, lemmatizer, stop_words))
    return df


def open_and_preprocess(filename):
    df = pd.read_csv(filename)    
    
    # create 3 new columns
    df['week_day'] = df['pub_date'].map(lambda x: pd.Timestamp.to_pydatetime(pd.Timestamp(x)).weekday())
    df['pub_hour'] = df['pub_date'].map(lambda x: pd.Timestamp.to_pydatetime(pd.Timestamp(x)).hour)
    df['all_text'] = df['headline'] + " " + df['abstract'] + " " + df['keywords']
    
    # ordinal encode
    df = ord_encode(df, ['newsdesk', 'section', 'material'])
    
    df = df.drop(['uniqueID', 'subsection', 'pub_date', 'headline', 'abstract', 'keywords'], axis=1)
    
    return df


def process_column(df, column, n_top, popular_words = []):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) 
    df[column] = df[column].astype(str)
    df = lemmatize_column(df, column, lemmatizer, stop_words)
    if len(popular_words) == 0:
        popular_words = get_popular_words(df[column], n_top)
    df = encode_language_column(df, column, popular_words)
    df = df.drop([column], axis=1)
    return df


### OPTIMIZE 
def base_optimize_DT(X_train, y_train):    
    # Cross-validation folds
    k = 10

    # Hyperparameters to tune:
    params = {
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10]
             }
    
    # Initialize GridSearchCV object with decision tree classifier and hyperparameters
    grid_tree = sklearn.model_selection.GridSearchCV(estimator=sklearn.tree.DecisionTreeClassifier(),
                             param_grid=params,
                             cv=k,
                             return_train_score=True,
                             scoring='accuracy',
                             refit='accuracy') 

    # Train and cross-validate, print results
    grid_tree.fit(X_train, y_train)

    best_hyperparams = grid_tree.best_params_

    # print best hyperparameters
    print("Optimized Decision Tree")
    print(best_hyperparams)
    return best_hyperparams


def base_optimize_KNN(X_train, y_train):
    ## Choose optimal k neighbors  
    # Cross-validation folds
    neighbors = []                                                              # Empty list to store neighbors
    cv_scores = []                                                              # Empty list to store scores

    # Perform 10-fold cross validation with K=5 for KNN (the n_neighbors parameter)
    for k in range(1, 51, 2):                                                   # Range of K we want to try
        neighbors.append(k) 
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = k)           # k = 5 for KNeighborsClassifier
        scores = sklearn.model_selection.cross_val_score( 
            knn, X_train, y_train, cv = 10, scoring = 'accuracy') 
        cv_scores.append(scores.mean()) 

    # Misclassification error versus k
    MSE = [1-x for x in cv_scores]                                               # Changing to misclassification error

    # Determining the best k value
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('The optimal number of K neighbors = %d ' %optimal_k)

    # Plot misclassification error versus k
    plt.figure(figsize = (10,6))
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of K neighbors')
    plt.ylabel('Misclassification Error')
    plt.show()
    plt.savefig('k_neighbors.png')
    plt.clf()

    # Hyperparameters to tune:
    params = {
                'n_neighbors': [19], 
                'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'weights': ['uniform', 'distance'],
             }
    
    # Initialize GridSearchCV object 
    grid_tree = sklearn.model_selection.GridSearchCV(estimator=sklearn.neighbors.KNeighborsClassifier(),
                             param_grid=params,
                             cv=10,
                             return_train_score=True,
                             scoring='accuracy',
                             refit='accuracy') 

    # Train and cross-validate, print results
    grid_tree.fit(X_train, y_train)

    best_hyperparams = grid_tree.best_params_

    # print best hyperparameters
    print("Optimized KNN")
    print(best_hyperparams)
    return best_hyperparams


def base_optimize_XGBM(X_train, y_train):    
    # Cross-validation folds
    k = 5

    # Hyperparameters to tune:
    params = {'booster': ['gblinear', 'gbtree', 'dart'],
              'n_estimators': [10, 25, 50],
              'max_depth': [5, 10, 20],
             }
    
    # Initialize GridSearchCV object
    grid_tree = sklearn.model_selection.GridSearchCV(estimator=xgb.XGBClassifier(),
                             param_grid=params,
                             cv=k,
                             return_train_score=True,
                             scoring='accuracy',
                             refit='accuracy') 

    # Train and cross-validate, print results
    grid_tree.fit(X_train, y_train)

    best_hyperparams = grid_tree.best_params_

    # print best hyperparameters
    print("Optimized XGBoost")
    print(best_hyperparams)
    return best_hyperparams


### PREPROCESSING [ALL TEXT]
n_top = 150
text_column_to_change = 'all_text'

# TRAIN SET 
df = open_and_preprocess("train.csv")
df = process_column(df, text_column_to_change, n_top)
y_train = df['is_popular']
# NOTE: REMOVING word_count DRASTICALLY IMPROVES ACCURACY
X_train = df.drop(['is_popular', 'n_comments', 'word_count'], axis=1)

# TEST SET 
ts = open_and_preprocess("test.csv")
ts = process_column(ts, text_column_to_change, n_top, get_pw_from_file('train.csv', text_column_to_change, n_top))
y_test = ts['is_popular']
X_test = ts.drop(['is_popular', 'word_count'], axis=1)


### MODEL TRAINING

## OPTIMIZE SHALLOW MODELS
base_optimize_DT(X_train, y_train)
base_optimize_KNN(X_train, y_train)
base_optimize_XGBM(X_train, y_train)


# PREDICTING WITH DECISION TREE
tree = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10, min_samples_split=2)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
dt_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"Decision Tree's test accuracy is {dt_accuracy}")


# PREDICTING WITH KNN
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=19, weights='distance', algorithm='kd_tree')
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
knn_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"KNN's test accuracy is {knn_accuracy}")


# PREDICTING WITH XGBoost
boost = xgb.XGBClassifier(booster='gbtree', max_depth=10, n_estimators=25, use_label_encoder=False)
boost.fit(X_train, y_train)
y_test_pred = boost.predict(X_test)
boost_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"XGBoost's test accuracy is {boost_accuracy}")


# PREDICTING WITH FNN
fnn = ks.models.Sequential()
fnn.add(ks.layers.Flatten(input_shape=[X_train.shape[1]]))
fnn.add(ks.layers.Dense(64, activation="relu"))
fnn.add(ks.layers.Dense(32, activation="relu"))
fnn.add(ks.layers.Dense(16, activation="relu"))
fnn.add(ks.layers.Dense(8, activation="relu"))
fnn.add(ks.layers.Dense(2, activation="softmax"))

fnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
fnn.fit(X_train, y_train, batch_size=50, epochs=20, validation_split=0.15)

test_predictions = np.argmax(fnn.predict(X_test), axis=1)
fnn_accuracy = sklearn.metrics.accuracy_score(y_test, test_predictions)
print(f"FNN's test accuracy is {fnn_accuracy}")


## MODEL ANALYSIS 

# Feature importance of Decision Tree 
feat_importances = pd.Series(tree.feature_importances_, index=X_train.columns).nlargest(10)
feat_importances.sort_values(ascending=True)
feat_importances.plot(kind='barh')
plt.ylabel('Features')
plt.title("Decision Tree's Feature Importance")
plt.show()
plt.savefig('dt_feature_importance.png')
plt.clf()


# Plotting KNN decision region
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=19)
X_train2 = TSNE(n_components = 2).fit_transform(X_train)
clf.fit(X_train2, y_train)
plot_decision_regions(X_train2, y_train.values, clf=clf, legend=2)
# Adding axes annotations
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN with K = 19')
plt.show()
plt.savefig('knn_surface.png')
plt.clf()


# Show feature importance 
xgb.plot_importance(boost, max_num_features=10)
plt.title("XGBoost Classifier's Feature Importance")
plt.show()
plt.savefig('feature_importance.png')
plt.clf()


Based on the test values generate the confusion matrix 
Summary of the predictions made by the classifier
mat = sklearn.metrics.confusion_matrix(y_test, test_predictions) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) 
plt.title("FNN's Confusion Matrix")
plt.xlabel('true class') 
plt.ylabel('predicted class')
plt.show()
plt.savefig('fnn_confusion_matrix.png')
plt.clf()


# Decision Tree's test accuracy is 0.6648310387984981
# KNN's test accuracy is 0.6868585732165207
# XGBoost's test accuracy is 0.7121401752190237
# FNN's test accuracy is 0.7531914893617021
# 218/218 [==============================] - 0s 1ms/step - loss: 0.3657 - accuracy: 0.8325 - val_loss: 0.5921 - val_accuracy: 0.7410


