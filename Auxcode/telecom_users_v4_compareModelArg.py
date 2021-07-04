import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc, plot_confusion_matrix, recall_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
    # plt.savefig(f"./resultados/{title}.png")

def to_label(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

#Global Variable
cv = 10

# Ler o ficheiro de dados
## Só tem um data Set. Secalhar devo arranjar outro
df = pd.read_csv("../dados/telecom_users.csv")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)  # Verificar o que é random_state

# Resolver a falta de balanceamento do dataset
SMOTE_oversample = SMOTE(random_state=42)
X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)

# name = "SVC"
# clf = SVC()
# clf_opt = SVC(kernel="linear", tol=0.001)
name = "MLPClassifier"
clf = MLPClassifier()
clf_opt = MLPClassifier(activation="relu", solver="adam", learning_rate="adaptive")


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
title = "toBe _ OPTIMIZED"

#Confusion Matrix
disp = plot_confusion_matrix(clf, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
disp.figure_.suptitle(f"{name}_{title}")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %\n")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f} %\n")
plt.show()

clf_opt.fit(X_train, y_train)
y_pred = clf_opt.predict(X_test)
title = "toBe _ OPTIMIZED"

#Confusion Matrix
disp = plot_confusion_matrix(clf_opt, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
disp.figure_.suptitle(f"{name}_{title}")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %\n")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f} %\n")
plt.show()