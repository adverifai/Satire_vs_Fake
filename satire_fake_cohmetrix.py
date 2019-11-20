from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import copy
import numpy as np
import pandas as pd
import my_utils
from pre_processing import text_clean
from my_utils import drop_constant_columns


def create_cohmetrix_input():
    """
    step [1]: creating single text files per document
    file name format: d_[index]_[label]
    label -> fake: 0, satire: 1
    :return:
    """
    data = my_utils.read_fake_satire_dataset("data/FakeNewsData/StoryText 2/")

    for index, row in data.iterrows():
        with open('data/cohmetrix/input/d' + str(row.doc_id) + '_' + str(row.bin_label) + '.txt', 'w+') as text_file:
            # if we want to clean the text
            # text_file.write(text_clean(row[0], True, True, False, 1))
            text_file.write(row.document)
            text_file.close()


def create_regresssion_input():
    """
    step [2]: creating a single excel file as input to regression analysis using the coh-metrix output
    satirefake.csv: the output of coh-metrix
    :return:
    """
    coh_data = pd.read_csv("data/cohmetrix/output/satirefake.csv")
    x_columns = list(coh_data.iloc[:, 1:])
    x_reg = []
    y_reg = []

    for item in coh_data.iterrows():
        tmp = item[1][0].split('\\')
        tmp = tmp[len(tmp)-1].split('.')[0].split('_')
        doc_label = int(tmp[1])

        x_reg.append(item[1][1:])
        y_reg.append(doc_label)

    # y_reg = np.array(y_reg)

    # converting list to dataframe
    x_reg = pd.DataFrame(np.array(x_reg).reshape(len(x_reg), len(x_columns)), columns=x_columns)
    # dropping constant value columns
    # x_lin = np.array(x_lin)
    x_reg = drop_constant_columns(x_reg)

    # scaling the data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_lin = scaler.fit_transform(x_lin)

    x_full = copy.deepcopy(x_reg)
    x_full["label"] = y_reg
    x_full = x_full.fillna(0)
    writer = pd.ExcelWriter('data/satire_fake_full.xlsx')
    x_full.to_excel(writer, 'Sheet1')
    writer.save()

    print("Created regression input files successfully.")


def logistic_regression(x, y, model_features):
    """
    non-cross validation version of logistic regression
    :param x:
    :param y:
    :param model_features:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    logreg = LogisticRegression(class_weight="balanced", solver='lbfgs')
    logreg.fit(x_train, y_train)
    print("\n\nLogistic regression report")
    print("=========================")
    if model_features != "":
        # print(logreg.coef_)
        feature_coeff = list(zip(logreg.coef_[0], model_features))
        for item in sorted(feature_coeff, key=lambda x: abs(x[0])):
            print(item)
    y_pred = logreg.predict(x_test)
    print('Accuracy: {:.2f}'.format(logreg.score(x_test, y_test)))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))


def my_classifier(x, y, clf):
    """
    binary classification using cross validation
    :param x: independent variables
    :param y: dependent variable
    :param clf: classifier
    :return:
    """
    scoring = {'f1': 'f1_micro', 'prec': 'precision', 'rec': 'recall'}
    scores = cross_validate(clf, x, y, cv=10, scoring=scoring)
    print("Precision (test): " + str(scores['test_prec'].mean()))
    print("Recall (test): " + str(scores['test_rec'].mean()))
    print("F1 (test): " + str(scores['test_f1'].mean()))

    # if only using one scoring metric
    # scores = cross_val_score(clf, x, y, cv=10, scoring=scoring)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


create_regresssion_input()

data = pd.read_csv("data/classification.csv")
model_features = list(data)

x = data.iloc[:, 1:len(model_features) - 1]
y = data.iloc[:, len(model_features) - 1]

model_features = list(x)
# naive bayes
print("Naive Bayes")
my_classifier(x, y, GaussianNB())
print("---------------------------")
# svm
print("SVM")
my_classifier(x, y, SVC(kernel='linear', C=1))
print("---------------------------")
# logistic regression
print("Logistic Regression")
my_classifier(x, y, LogisticRegression(class_weight="balanced", solver='lbfgs'))
print("---------------------------")
print("Gradient Boosting")
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(learning_rate=learning_rate)
    my_classifier(x, y, gb)
