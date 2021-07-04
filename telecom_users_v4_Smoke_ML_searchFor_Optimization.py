##
# TEMA 1 - Telecom Users Dataset
# Treino utilizando Machine Learning (sklearn)
# Pesquisa dos melhores argumentos de cada algoritmo.
# Contém a comparação antes e depois de cada optimização
##

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc, plot_confusion_matrix, recall_score, precision_score
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
    "GaussianProcessClassifier", #GaussianProcessClassifier(),
    "GaussianNB", #GaussianNB(),
    "LogisticRegression", #LogisticRegression(),
    "KNeighborsClassifier", #KNeighborsClassifier(),
    "DecisionTreeClassifier", # DecisionTreeClassifier(),
    "Random Forest", #  RandomForestClassifier(),
    "RadiusNeighborsClassifier", #RadiusNeighborsClassifier(radius=4.0),
    "MLPClassifier", #MLPClassifier(),
    "Gradient Boost", #GradientBoostingClassifier(),
]

classifiers = [
    SVC(),
    GaussianProcessClassifier(),
    GaussianNB(),
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    RadiusNeighborsClassifier(radius=4.0),
    MLPClassifier(),
    GradientBoostingClassifier()
]

for i, (name, clf) in enumerate(zip(classifier_names, classifiers)):
    file = f"./resultados/_ResultsV4/_Result_AfterOptimization2/matrixConfusion_{name}.txt"
    if os.path.exists(file):
        os.remove(file)

    f = open(file, "a")

    print(f"\nClassifier: {name}\n")

    # Split DataSet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)  # Verificar o que é random_state

    # Resolver a falta de balanceamento do dataset
    SMOTE_oversample = SMOTE(random_state=42)
    X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    title = "toBe _ OPTIMIZED"

    # Tabela summary
    print(classification_report(y_test, y_pred))

    #Confusion Matrix
    disp = plot_confusion_matrix(clf, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
    disp.figure_.suptitle(f"{name}_{title}")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.savefig(f"./resultados/_ResultsV4/_Result_AfterOptimization2/matrixConfusion_{name}.png")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f} %\n", file=f)
    print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f} %\n", file=f)
    plt.show()

    parameters = [
        [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'tol': [1e-1, 1e-3, 1e-4]}], # SVC
        [{'max_iter_predict':[100, 125]}],  # GaussianProcessClassifier
        [{'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12]}],  # GaussianNB
        [{'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'multi_class': ['auto', 'ovr', 'multinomial']}],  # LogisticRegression
        [{'weights':['uniform', 'distance']}],  # KNeighborsClassifier
        [{'criterion': ['gini', 'entropy'], 'splitter':['best', 'random']}],  # DecisionTreeClassifier
        [{'n_estimators': [10, 50, 100, 150, 200, 400, 1000], 'max_features': ['auto', 'log2']}],  # Random Forest
        [{'radius': [1, 2, 4, 5, 6, 7, 8], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [30, 40, 50, 60, 70, 80]}],  # RadiusNeighborsClassifier
        [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs', 'sgd', 'adam'], 'tol':[1e-1, 1e-3, 1e-4]}],  # MLPClassifier
        [{'tol':[1e-1, 1e-3, 1e-4], 'learning_rate':[0.1, 0.2, 0.3, 0.4]}],  # Gradient Boost
    ]

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=parameters[i],
                               scoring='recall',
                               cv=cv,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_recall = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print(f"\nBest Recall: {best_recall * 100:.2f} %")
    print(f"Best Parameters: {best_parameters}\n")

    result = []
    clf2 = ""
    for key in best_parameters:
        result.append(best_parameters[key])

    print(result)
    print(f"opções optimizadas: {result} %\n", file=f)

    if i == 0:
        clf2 = SVC(kernel=result[0], tol=result[1])
    elif i == 1:
        clf2 = GaussianProcessClassifier(max_iter_predict=result[0])
    elif i == 2:
        clf2 = GaussianNB(var_smoothing=result[0])
    elif i == 3:
        clf2 = LogisticRegression(multi_class=result[0], penalty=result[1], solver=result[2])
    elif i == 4:
        clf2 = KNeighborsClassifier(weights=result[0])
    elif i == 5:
        clf2 = DecisionTreeClassifier(criterion=result[0], splitter=result[1])
    elif i == 6:
        clf2 = RandomForestClassifier(max_features=result[0], n_estimators=result[1])
    elif i == 7:
        clf2 = RadiusNeighborsClassifier(algorithm=result[0], leaf_size=result[1], radius=result[2], n_jobs=-1)
    elif i == 8:
        clf2 = MLPClassifier(activation=result[0], solver=result[1], tol=result[2])
    elif i == 9:
        clf2 = GradientBoostingClassifier(tol=result[0], learning_rate=result[1])

    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_test)
    title = "OPTIMIZED"

    # Confusion Matrix
    disp = plot_confusion_matrix(clf2, X_test, y_test, display_labels=['Ficou(0)', 'Saiu(1)'])
    disp.figure_.suptitle(f"{name}_{title}")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.savefig(f"./resultados/_ResultsV4/_Result_AfterOptimization2/matrixConfusion_{name}.png")
    print(f"Precision _ Optimized: {precision_score(y_test, y_pred) * 100:.2f} %\n", file=f)
    print(f"Recall _ Optimized: {recall_score(y_test, y_pred) * 100:.2f} %\n", file=f)
    plt.show()