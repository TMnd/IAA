##
# TEMA 1 - Telecom Users Dataset
# -----------------  Não utilizado!!! -----------------
##

# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
from collections import Counter

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, f1_score, plot_confusion_matrix
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
from imblearn.combine import SMOTETomek
from pandas_profiling import ProfileReport

# import keras
# from keras.layers import Dense
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential

def confusion_matrix2(algorithm, X_test, y_test, title):
    plot_confusion_matrix(algorithm, X_test, y_test, cmap='Blues')
    plt.grid(False)
    plt.title(title)
    plt.show()

def tableCVData(clf, X_test, y_test, y_prob, cv, X_train, y_train):
    print(f'ROC AUC score: {round(roc_auc_score(y_test, y_prob), 3)}')
    print('-----------------------------------------------------')
    print('Valores medios dos scores de Cross-validation com 5 "folds" :\n')
    # print(f"ROC AUC: {round(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc').mean(), 3)}")
    print(f"precision: {round(cross_val_score(clf, X_train, y_train, cv=cv, scoring='precision').mean(), 2)}")
    print(f"recall: {round(cross_val_score(clf, X_train, y_train, cv=cv, scoring='recall').mean(), 2)}")
    print(f"f1: {round(cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean(), 2)}")

def roc_auc(y_test,y_prob,title):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    sns.set_theme(style='white')
    plt.figure(figsize=(8, 8))
    plt.plot(false_positive_rate, true_positive_rate, color='#b01717', label='AUC = %0.3f' % roc_auc)
    # plt.plot(a, color='#b01717', label='threshold = %0.3f' % a)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [a, 1], linestyle='--', color='#174ab0')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#174ab0')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.show()
    plt.savefig(f"./resultados/{title}.png")

def to_label(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# Ler o ficheiro de dados
## Só tem um data Set. Secalhar devo arranjar outro
df = pd.read_csv("dados/telecom_users.csv")



# profile = ProfileReport(df, title="Pandas Profiling Report")
# profile.to_file("report.templates")

# Reduzir os dados às colunas relevantes (para o que queremos fazer) e sanitar os valores
columns = ['Unnamed: 0', 'customerID', 'Dependents', 'PaperlessBilling', 'PaymentMethod']
df.drop(columns, inplace=True, axis=1)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'Yes': 1, 'No': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['OnlineBackup'] = df['OnlineBackup'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['DeviceProtection'] = df['DeviceProtection'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['TechSupport'] = df['TechSupport'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['StreamingTV'] = df['StreamingTV'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['StreamingMovies'] = df['StreamingMovies'].map({'No internet service': 0, 'No': 0, 'Yes': 1})
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.NaN, regex=True)
df['TotalCharges'].fillna(-1, inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'] = df['TotalCharges'].replace(-1, df['TotalCharges'].mean())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['tenure'] = df['tenure'].replace(0, df['tenure'].mean())

# METRICA EXTRAS
## One-hot encoding
### Do tipo de contratos
encoded_columns = pd.get_dummies(df['Contract'])
df = df.join(encoded_columns).drop('Contract', axis=1)
encoded_columns2 = pd.get_dummies(df['InternetService'])
df = df.join(encoded_columns2).drop('InternetService', axis=1)
## Media dos serviços incritos.
mean_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['avg'] = df[mean_cols].mean(axis=1)

df['log+1'] = (df['MonthlyCharges']+1).transform(np.log)
df.drop(['MonthlyCharges'], 1)

print(f"Number of features: {len(df.columns)}")

# Colocar os dados (os exemplos) na variável X e as respostas na variável y
X = np.array(df.drop(['Churn'], 1))
X = preprocessing.scale(X)  # ... normalização e standarização ... passar os dados (exemplos) para a escala [-1;1]
y = np.array(df['Churn'])

print("Numero de dados: ")
counter = Counter(y)
print(counter)

# Split DataSet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #Verificar o que é random_state

# Cross-validation
cv = KFold(n_splits=5)
cv.get_n_splits(X_train)

# # Resolver a falta de equilibrio do dataset
# SMOTE_oversample = SMOTE(random_state=42)
# # SMOTE_oversample = BorderlineSMOTE()
# # SMOTE_oversample = SMOTENC(categorical_features=[2, 13], random_state=42)
# # SMOTE_oversample = SMOTETomek(random_state=42)
# # X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train.ravel())
# X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)

for lr in range(20):
    # Split DataSet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # Verificar o que é random_state

    if lr == 0:
        clf = LogisticRegression(penalty="l1", solver='liblinear') #Done
        plotTitle = "LogisticRegression _ com SMOTE"
    elif lr == 1:
        clf = LogisticRegression(penalty="l1", solver='liblinear')  # Done
        plotTitle = "LogisticRegression _ sem SMOTE"
    elif lr == 2:
        clf = LogisticRegression(penalty="l1", solver='liblinear')  # Done
        plotTitle = "LogisticRegression com threshold _ com SMOTE"
    elif lr == 3:
        clf = LogisticRegression(penalty="l1", solver='liblinear')  # Done
        plotTitle = "LogisticRegression com threshold _ sem SMOTE"
    elif lr == 4:
        clf = LogisticRegressionCV(penalty="l1", cv=cv, random_state=0, solver="liblinear", n_jobs=-1)
        plotTitle = "LogisticRegression CV _ com SMOTE"
    elif lr == 5:
        clf = LogisticRegressionCV(penalty="l1", cv=cv, random_state=0, solver="liblinear", n_jobs=-1)
        plotTitle = "LogisticRegression CV _ sem SMOTE"
    elif lr == 6:
        clf = LogisticRegressionCV(penalty="l1", cv=cv, random_state=0, solver="liblinear", n_jobs=-1)
        plotTitle = "LogisticRegression CV com threshold _ com SMOTE"
    elif lr == 7:
        clf = LogisticRegressionCV(penalty="l1", cv=cv, random_state=0, solver="liblinear", n_jobs=-1)
        plotTitle = "LogisticRegression CV com threshold _ sem SMOTE"
    elif lr == 8:
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss="log"))
        plotTitle = "LogisticRegression SGDClassifier _ com SMOTE"
    elif lr == 9:
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss="log"))
        plotTitle = "LogisticRegression SGDClassifier _ sem SMOTE"
    elif lr == 10:
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss="log"))
        plotTitle = "LogisticRegression SGDClassifier com threshold _ com SMOTE"
    elif lr == 11:
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss="log"))
        plotTitle = "LogisticRegression SGDClassifier com threshold _ sem SMOTE"
    elif lr == 12:
        clf = KNeighborsClassifier(n_neighbors=10)  # O melhor é o n=10 com o "algorithm" em auto
        plotTitle = "KNeighborsClassifier _ com SMOTE"
    elif lr == 13:
        clf = KNeighborsClassifier(n_neighbors=10)  # O melhor é o n=10 com o "algorithm" em auto
        plotTitle = "KNeighborsClassifier _ sem SMOTE"
    elif lr == 14:
        clf = KNeighborsClassifier(n_neighbors=10)  # O melhor é o n=10 com o "algorithm" em auto
        plotTitle = "KNeighborsClassifier com threshold _ com SMOTE"
    elif lr == 15:
        clf = KNeighborsClassifier(n_neighbors=10)  # O melhor é o n=10 com o "algorithm" em auto
        plotTitle = "KNeighborsClassifier com threshold _ sem SMOTE"
    elif lr == 16:
        DT = tree.DecisionTreeClassifier(max_features=5, max_depth=10, min_samples_split=90, min_samples_leaf=30,random_state=1)
        plotTitle = "DecisionTreeClassifier _ com SMOTE"
    elif lr == 17:
        DT = tree.DecisionTreeClassifier(max_features=5, max_depth=10, min_samples_split=90, min_samples_leaf=30,random_state=1)
        plotTitle = "DecisionTreeClassifier _ sem SMOTE"
    elif lr == 18:
        DT = tree.DecisionTreeClassifier(max_features=5, max_depth=10, min_samples_split=90, min_samples_leaf=30,random_state=1)
        plotTitle = "DecisionTreeClassifier com threshold _ com SMOTE"
    elif lr == 19:
        DT = tree.DecisionTreeClassifier(max_features=5, max_depth=10, min_samples_split=90, min_samples_leaf=30,random_state=1)
        plotTitle = "DecisionTreeClassifier com threshold _ sem SMOTE"

    print(f"\n{plotTitle}\n")

    if lr%2 == 0:
        # Resolver a falta de equilibrio do dataset
        SMOTE_oversample = SMOTE(random_state=42)
        X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)
        counter = Counter(y_train)
        print(f"---Numero de dados apos o Smote: {counter}")

    clf.fit(X_train, y_train)
    #train
    predict_train = clf.predict(X_train)
    predict_train_prob = clf.predict_proba(X_train)  # [:, 1]
    # print(predict_train_prob)
    # print(predict_train)
    # print('-- train: ', predict_train)
    #test
    predict_test = clf.predict(X_test)
    predict_test_prob = clf.predict_proba(X_test)  # [:, 1]
    # print(predict_test_prob)
    # print(predict_test)
    # print('-- teste: ', LogisticRegression_test_score)
    # accuracy score
    LogisticRegression_train_score = clf.score(X_train, y_train)
    LogisticRegression_test_score = clf.score(X_test, y_test)
    print('Accuracy(Score) no treino: ', LogisticRegression_train_score)
    print('Accuracy(Score) no teste: ', LogisticRegression_test_score)
    # f1-score
    LogisticRegression_f1_score = f1_score(y_test, predict_test)
    print('F1-score no teste:', LogisticRegression_f1_score)
    # DADOS PARA DEMONSTRAR
    if lr == 2 or lr == 3 or lr == 6 or lr == 7 or lr == 10 or lr == 11 or lr == 14 or lr == 15 or lr == 18 or lr == 19:
        probs = predict_test_prob[:, 1]  # predict_train_prob
        threshold = np.arange(0,1,0.0001)
        scores = [f1_score(y_test, to_label(probs, t)) for t in threshold]
        it = np.argmax(scores)
        # print(scores)
        print('Best Threshold=%f' % (threshold[it]))
        predicted_labels_set = to_label(probs, threshold[it])
        classification_report_variable2 = predict_test
        congusionMatrix_variable2 = predicted_labels_set
        print(congusionMatrix_variable2)
        roc_variable2 = predicted_labels_set
    else:
        classification_report_variable2 = predict_test
        congusionMatrix_variable2 = predict_test
        roc_variable2 =predict_test_prob[:, 1]
    print(classification_report(y_test, classification_report_variable2))
    cm = confusion_matrix(y_test, congusionMatrix_variable2)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Ficou', 'Saiu'])
    cmd.plot()
    plt.savefig(f'./resultados/{plotTitle.split(" ")[0]}/{plotTitle}.png')
    roc_auc(y_test, roc_variable2, plotTitle)
    print("---------------------------------------------")

#Neural network models :: Classification
MLPClassifier = MLPClassifier(random_state=0, max_iter=200)
# MLPClassifier = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
plotTitle = "MLPClassifier"
MLPClassifier.fit(X_train, y_train)
# model score
predict_train_MLPClassifier = MLPClassifier.predict(X_train)
predict_test_MLPClassifier = MLPClassifier.predict(X_test)
# accuracy score
MLPClassifier_train_score = MLPClassifier.score(X_train, y_train)
MLPClassifier_test_score = MLPClassifier.score(X_test, y_test)
print('Accuracy on Train set', MLPClassifier_train_score)
print('Accuracy on Test set', MLPClassifier_test_score)
# f1-score
MLPClassifier_f1_score = f1_score(y_test, predict_test_MLPClassifier)
print('F1-score on Test set:', MLPClassifier_f1_score)
# DADOS PARA DEMONSTRAR
print(classification_report(y_test, predict_test_MLPClassifier))
MLPClassifier_test_score = MLPClassifier.score(X_test, y_test)
# confusion_matrix2(MLPClassifier, X_test, y_test, plotTitle)
# confusion_matrix2(MLPClassifier, X_train, y_train, plotTitle)
y_prob = MLPClassifier.predict_proba(X_test)[:, 1]
tableCVData(MLPClassifier, X_test, y_test, y_prob, cv, X_train, y_train)
print('Valores medios dos scores de Cross-validation com 5 "folds" :\n')
print(f"ROC AUC: {round(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc').mean(), 3)}")

#---
# print(f"precision: {round(cross_val_score(MLPClassifier, X_train, y_train, cv=cv, scoring='precision').mean(), 2)}")
# print(f"recall: {round(cross_val_score(MLPClassifier, X_train, y_train, cv=cv, scoring='recall').mean(), 2)}")
# print(f"f1: {round(cross_val_score(MLPClassifier, X_train, y_train, cv=cv, scoring='f1').mean(), 2)}")

# roc_auc(y_test, y_prob)
# print("---------------------------------------------")
