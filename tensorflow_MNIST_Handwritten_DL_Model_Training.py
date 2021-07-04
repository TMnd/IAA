##
# TEMA 2 - MNIST hand written digit database
# Treino CNN utilizando Deep Learning (TensorFlow)
##

import PIL
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import utils
from sklearn.metrics import confusion_matrix
from externalFunctions import model_handWriting_numbers, normalize_image
from sklearn.model_selection import train_test_split

epochs=100
batch_size=200

def load_data(path):
    print(path)
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


#--------
## option2
# df = pd.read_csv("E:\AII\Aula 3 - 26-05-2021\dados\mnist_train_WithSymbols.csv", delimiter=',')
# df = df.to_numpy()
# np.save('dados/mnist_train_WithSymbols', df)
# (X_train, y_train),(X_test, y_test) = load_data("dados\mnist_train_WithSymbols.npy")

## option1
df = pd.read_csv("dados/mnist_train_WithSymbols_Balance_v2.csv")

df.head()

# Colocar os dados (os exemplos) na variável X e as respostas na variável y
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #Verificar o que é random_state

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize the train dataset
X_train = tf.keras.utils.normalize(X_train, axis=1)
# Normalize the test dataset
X_test = tf.keras.utils.normalize(X_test, axis=1)
# one hot encode outputs
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

num_classes = y_test.shape[1]

rounded_labels=np.argmax(y_test, axis=1)

checkpoint_path = "savedModels/handWriting_WithPlusSymbol_balance2_2_With()_100/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

print(num_classes)

model = model_handWriting_numbers(num_classes)

model.summary()

model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(x=X_train, y=y_train, validation_split=0.25, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

rounded_predictions = model.predict_classes(X_test, batch_size=batch_size, verbose=0)

confusion_mtx = confusion_matrix(rounded_labels, rounded_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=X_test, y=y_test)
print('\nTest accuracy:', test_acc)

# test single images
# imagem de teste singular
img = PIL.Image.open(pathlib.Path("ficheirosParaTestar/3.png")).convert('L').resize((28, 28), PIL.Image.ANTIALIAS)
img = np.array(img)
img = normalize_image(img)
plt.imshow(img, cmap="gray")
plt.show()

my_tests = np.array([img])
predictions = model.predict([my_tests])
print("Prediction value:", np.argmax(predictions[0]))