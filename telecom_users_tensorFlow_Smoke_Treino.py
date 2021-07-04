##
# TEMA 1 - Telecom Users Dataset
# Treino utilizando Deep Learning (TensorFlow)
# Pesquisa de soluções:
#   https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
#   https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
##

from collections import Counter

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from IPython.display import display
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
    plt.savefig(f"./resultados/TensorFlow/{title}.png")

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

#Neural network Tenserflow.
# Dividir o conjunto de dados para o subconjunto de treino e subconjunto de teste (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Create training and validation splits
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_valid = df_valid.drop('Churn', axis=1)
y_valid = df_valid['Churn']

# Resolver a falta de equilibrio do dataset
SMOTE_oversample = SMOTE(random_state=42)
X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train)

# Normalização dos dados da entrada dos conjuntos de treino e teste
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(len(X_train))
print(len(y_train))

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

#https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev#:~:text=Choosing%20Hidden%20Layers&text=If%20data%20is%20less%20complex,hidden%20layers%20can%20be%20used.
#sqrt(input layer nodes * output layer nodes)

#Build the network
model = Sequential([
    # the hidden ReLU layers
    layers.Dense(5, activation = 'relu', input_dim = 22),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(5, activation = 'relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(5, activation = 'relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    # the linear output layer
    layers.Dense(1, activation = 'sigmoid')
])

#loss function
model.compile(
    optimizer="adam",
    loss = 'binary_crossentropy',
    #loss="mae",
    # metrics = ['accuracy']
    metrics = ['binary_accuracy']
    #metrics = ['Recall']
)

# model.fit(X_train, y_train, epochs = 100)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=10,
    epochs=250,
    # callbacks=[early_stopping],  # put your callbacks in a list
    # verbose=1,  # turn off training log
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)#"[:, 1]
y_pred = (y_pred > 0.35)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Ficou', 'Saiu'])
cmd.plot()
# plt.savefig("./resultados/TensorFlow/confusionMatrix.png")
plt.show()

roc_auc(y_test, y_prob, "TensorFlow")

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
# plt.savefig(f'./resultados/TensorFlow/loss_valLoss.png')
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
# plt.savefig(f'./resultados/TensorFlow/binary_accuracy__val_binary_accuracy.png')

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(),
              history_df['val_binary_accuracy'].max()))
plt.show()

