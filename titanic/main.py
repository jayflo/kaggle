"""
A starter script for working on the titanic survivor prediction competition.
Written by following dataquest.com tutorial.
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import cleandata as clndt
import features as ftrs

# load
df = pd.read_csv('../data/titanic/train.csv')
clndt.clean(df)

# test
# df_test = pd.read_csv('../data/titanic/test.csv')

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# linear regression
alg = LinearRegression()
kf = KFold(df.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (df[predictors].iloc[train,:])
    train_target = df['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(df[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
accuracy = len(df.ix[df.Survived == predictions]) / len(df.index)

# alternatively...
# scores = cross_validation.cross_val_score(alg, df[predictors], df['Survived'], cv = 3)
# accuracy = scores.mean()

print('Linear regression train accuracy = ', accuracy)

# logistic regression
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, df[predictors], df['Survived'], cv=3)
print('Logistic regression train accuracy = ', scores.mean())

# random forest

alg = RandomForestClassifier(
    random_state=1,
    n_estimators=10,
    min_samples_split=2,
    min_samples_leaf=1
)
scores = cross_validation.cross_val_score(alg, df[predictors], df['Survived'], cv=3)
print('Random forest train accuracy = ', scores.mean())

# add features
ftrs.family_size(df) # FamilySize
ftrs.name_length(df) # NameLength
ftrs.title(df) # Title
ftrs.family_id(df) # FamilyId, requires FamilySize

# select best features
predictors = predictors + ['FamilySize', 'Title', 'FamilyId']
selector = SelectKBest(f_classif, k=5)
selector.fit(df[predictors], df['Survived'])
scores = -np.log10(selector.pvalues_)
print('\n')
print(predictors, scores)
print('\n')

# improved random forest
predictors = ['Pclass', 'Sex', 'Fare', 'Title']

alg = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=8,
    min_samples_leaf=4
)
scores = cross_validation.cross_val_score(alg, df[predictors], df['Survived'], cv=3)
print('Improved random forest train accuracy = ', scores.mean())

# gradient boosting
predictors_gbc = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title', 'FamilyId']
predictors_l = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
algos = [
    [GradientBoostingClassifier(
        random_state=1,
        n_estimators=25,
        max_depth=3
    ), predictors_gbc],
    [LogisticRegression(random_state=1), predictors_l]
]

kf = KFold(df.shape[0], n_folds=3, random_state=1)
predictions = []

for train, test in kf:
    train_target = df['Survived'].iloc[train]
    full_test_predictions = []

    for alg, predictors in algos:
        alg.fit(df[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(df[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions < 0.5] = 0
    test_predictions[test_predictions >= 0.5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = len(df.ix[df.Survived == predictions]) / len(df.index)
print(accuracy)

# kaggle submission
# submission = pd.DataFrame({
#     'PassengerId': df_test['PassengerId'],
#     'Survived': test_predictions
# })