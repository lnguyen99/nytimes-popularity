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
    print(top_words)
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
    params = {'min_samples_split': [2, 5, 10],
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
    # Cross-validation folds
    k = 10

    # Hyperparameters to tune:
    params = {'n_neighbors': [3, 5, 8, 10, 15],
                'weights': ['uniform', 'distance'],
             }
    
    # Initialize GridSearchCV object with decision tree classifier and hyperparameters
    grid_tree = sklearn.model_selection.GridSearchCV(estimator=sklearn.neighbors.KNeighborsClassifier(),
                             param_grid=params,
                             cv=k,
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
    
    # Initialize GridSearchCV object with decision tree classifier and hyperparameters
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

# OPTIMIZE MODEL
dt = base_optimize_DT(X_train, y_train)
knn = base_optimize_KNN(X_train, y_train)
boost = base_optimize_XGBM(X_train, y_train)

# PREDICTING WITH DECISION TREE
# sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10, min_samples_split=2)
dt.fit(X_train, y_train)
y_test_pred = dt.predict(X_test)
dt_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"Decision Tree's test accuracy is {dt_accuracy}")

# PREDICTING WITH KNN
# sklearn.neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
knn_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"KNN's test accuracy is {knn_accuracy}")

# PREDICTING WITH XGBoost
boost.fit(X_train, y_train)
y_test_pred = boost.predict(X_test)
boost_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
print(f"XGBoost's test accuracy is {boost_accuracy}")

# PREDICTING WITH FNN
fnn = ks.models.Sequential()
fnn.add(ks.layers.Flatten(input_shape=[shape]))
fnn.add(ks.layers.Dense(64, activation="relu"))
fnn.add(ks.layers.Dense(32, activation="relu"))
fnn.add(ks.layers.Dense(16, activation="relu"))
fnn.add(ks.layers.Dense(8, activation="relu"))
fnn.add(ks.layers.Dense(2, activation="softmax"))

fnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
fnn.fit(X_train, y_train, batch_size=128, epochs=30, validation_split=0.1)
test_predictions = np.argmax(fnn.predict(X_test), axis=1)
fnn_accuracy = sklearn.metrics.accuracy_score(y_test, test_predictions)
print(f"FNN's test accuracy is {fnn_accuracy}")

# Show feature importance 
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
plt.savefig('feature_importance.png')
plt.clf()

# Show DT 
tree.plot_tree(dt);
plt.savefig('dt_tree.png')
plt.clf()

# Show xgb tree
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()
plt.savefig('xgb_tree.png')
plt.clf()

# Show FNN model 
plot_model(fnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.savefig('fnn_model.png')
plt.clf()

## change the knn optimize 
## add knn decision surface 
## add confusion matrix
