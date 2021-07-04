##
# TEMA 1 - Telecom Users Dataset
# Treino utilizando Machine Learning (sklearn)
# Treino de varios algoritmos Machine Learning com os argumentos base e optimizados e calculo das varias métricas.
##

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, f1_score, plot_confusion_matrix, accuracy_score, recall_score,precision_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def roc_auc(y_test,y_prob,title,op):
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
    # plt.savefig(f"./resultados/{title}.png")
    if op == 0:
        plt.savefig(f"./resultados/_ResultsV4/{name}/roc_auc_{title}.png")
    else:
        plt.savefig(f"./resultados/_ResultsV4/{name}/roc_auc_{title}_optimize.png")

def to_label(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

#Global Variable
cv = 10

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

classifier_names = [
    "SVC", #SVC(),
    "GaussianNB", #GaussianNB(),
    "LogisticRegression", #LogisticRegression(),
    "SGDClassifier", #make_pipeline(StandardScaler(), SGDClassifier()),
    "KNeighborsClassifier", #KNeighborsClassifier(),
    "DecisionTreeClassifier", # DecisionTreeClassifier(),
    "Random Forest", #  RandomForestClassifier(),
    "RadiusNeighborsClassifier", #RadiusNeighborsClassifier(radius=4.0),
    "MLPClassifier", #MLPClassifier(),
    "Gradient Boost", #GradientBoostingClassifier(),
    "XGBoost", #XGBClassifier(use_label_encoder=False,eval_metric="logloss"),
    "GaussianProcessClassifier", #GaussianProcessClassifier(),
]

classifiers = [
    SVC(),
    GaussianNB(),
    LogisticRegression(),
    make_pipeline(StandardScaler(), SGDClassifier()),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    RadiusNeighborsClassifier(radius=4.0,), #Dá erro quando nao tem argumentos
    MLPClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    GaussianProcessClassifier(),
]

classifiers_optimized = [
    SVC(kernel="linear", tol=0.001),
    GaussianNB(var_smoothing=1e-09),
    LogisticRegression(multi_class="auto", penalty="l1", solver='liblinear'),
    make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss="log")),
    KNeighborsClassifier(weights="uniform", n_neighbors=10),
    DecisionTreeClassifier(max_features=5, max_depth=10, min_samples_split=90, min_samples_leaf=30),
    RandomForestClassifier(max_features="log2", n_estimators=400),
    RadiusNeighborsClassifier(algorithm="auto", leaf_size=30, radius=4.0, n_jobs=-1),
    MLPClassifier(activation="relu", solver="adam", learning_rate="adaptive"),
    GradientBoostingClassifier(tol=0.1, learning_rate=0.1),
    XGBClassifier(use_label_encoder=False,eval_metric="logloss"),
    GaussianProcessClassifier(max_iter_predict=100),
]


for name, clf, clf_optimize in zip(classifier_names, classifiers, classifiers_optimized):
    for i in range(2): #0 - Normal, 1 - Smote, 2 - Treshold
        print(f"\nClassifier: {name}\n")

        # Split DataSet
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
        if i == 1:
            # Resolver a falta de balanceamento do dataset
            SMOTE_oversample = SMOTE(random_state=42)
            X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            clf_optimize.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_optimize = clf_optimize.predict(X_test)
            roc_variable2 = y_pred
            roc_variable2_optimize = y_pred_optimize
            title = "SMOTE"
        # elif i == 2: //Não é usado
        #     clf.fit(X_train, y_train)
        #     y_pred = clf.predict(X_test)
        #     predict_test_prob = clf.predict_proba(X_test)  # [:, 1]
        #     probs = predict_test_prob[:, 1]  # predict_train_prob
        #     threshold = np.arange(0, 1, 0.0001)
        #     scores = [f1_score(y_test, to_label(probs, t)) for t in threshold]
        #     it = np.argmax(scores)
        #     # print(scores)
        #     print('Best Threshold=%f' % (threshold[it]))
        #     predicted_labels_set = to_label(probs, threshold[it])
        #     congusionMatrix_variable2 = predicted_labels_set
        #     print(congusionMatrix_variable2)
        #     roc_variable2 = predicted_labels_set
        else:
            clf.fit(X_train, y_train)
            clf_optimize.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_optimize = clf_optimize.predict(X_test)
            roc_variable2 = y_pred
            roc_variable2_optimize = y_pred_optimize
            title = ""

        # Tabela summary
        print("Default Model:")
        print(classification_report(y_test, y_pred))
        print("Optimize Model:")
        print(classification_report(y_test, y_pred_optimize))

        #Confusion Matrix
        # default model
        disp = plot_confusion_matrix(clf, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
        disp.figure_.suptitle(f"{name}_{title}")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.savefig(f"./resultados/_ResultsV4/{name}/matrixConfusion_{title}.png")
        plt.show()
        #optimize model
        disp = plot_confusion_matrix(clf_optimize, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
        disp.figure_.suptitle(f"{name}_{title}_optimize")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.savefig(f"./resultados/_ResultsV4/{name}/matrixConfusion_{title}_optimize.png")
        plt.show()

        # roc_auc
        roc_auc(y_test, roc_variable2, f"{name}_{title}", 0)
        roc_auc(y_test, roc_variable2_optimize, f"{name}_{title}_optimize", 1)

        if os.path.exists(f"./resultados/_ResultsV4/{name}/output_{title}.txt"):
            os.remove(f"./resultados/_ResultsV4/{name}/output_{title}.txt")

        f = open(f"./resultados/_ResultsV4/{name}/output_{title}.txt", "a")

        print(f"Default Model",file=f)
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %", file=f)
        print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f} %", file=f)
        print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f} %", file=f)
        print(f"F1: {f1_score(y_test, y_pred) * 100:.2f} %", file=f)

        print(f"\nOptimized Model", file=f)
        print(f"Accuracy: {accuracy_score(y_test, y_pred_optimize) * 100:.2f} %", file=f)
        print(f"Recall: {recall_score(y_test, y_pred_optimize) * 100:.2f} %", file=f)
        print(f"Precision: {precision_score(y_test, y_pred_optimize) * 100:.2f} %", file=f)
        print(f"F1: {f1_score(y_test, y_pred_optimize) * 100:.2f} %", file=f)

        # print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
        # print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f} %")
        # print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f} %")
        # print(f"F1: {f1_score(y_test, y_pred) * 100:.2f} %")

        print(f"\n", file=f)

        #k-Fold Cross Validation (with k=10)
        accuracy_array = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv)
        print(f"Accuracy (mean): {accuracy_array.mean() * 100:.2f} %", file=f)
        print(f"Accuracy (Standard Deviation): {accuracy_array.std() * 100:.2f} %\n", file=f)
        # print(f"Accuracy (mean): {accuracy_array.mean() * 100:.2f} %", file=f)
        # print(f"Accuracy (Standard Deviation): {accuracy_array.std() * 100:.2f} %\n", file=f)
        recall = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv, scoring='recall')
        print(f"Recall (mean): {recall.mean() * 100:.2f} %", file=f)
        print(f"Recall (Standard Deviation): {recall.std() * 100:.2f} %\n", file=f)
        # print(f"Recall (mean): {recall.mean() * 100:.2f} %", file=f)
        # print(f"Recall (Standard Deviation): {recall.std() * 100:.2f} %\n", file=f)
        precision = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv, scoring='precision')
        print(f"Precision (mean): {precision.mean() * 100:.2f} %", file=f)
        print(f"Precision (Standard Deviation): {precision.std() * 100:.2f} %\n", file=f)
        # print(f"Precision (mean): {precision.mean() * 100:.2f} %", file=f)
        # print(f"Precision (Standard Deviation): {precision.std() * 100:.2f} %\n", file=f)
        f1 = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv, scoring='f1')
        print(f"F1 (mean): {f1.mean() * 100:.2f} %", file=f)
        print(f"F1 (Standard Deviation): {f1.std() * 100:.2f} %\n", file=f)
        # print(f"F1 (mean): {f1.mean() * 100:.2f} %", file=f)
        # print(f"F1 (Standard Deviation): {f1.std() * 100:.2f} %\n", file=f)

        f.close()

    print("-------------------------------")
    # input("Press ENTER to continue...")