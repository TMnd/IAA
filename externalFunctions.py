##
# Funções externas para serem usadas
##

from sklearn.metrics import plot_confusion_matrix, roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np

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

def roc_auc(y_test, y_prob, title):
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
    #plt.savefig(f"./resultados/{title.split(' ')[0]}/roc_auc-{title}.png")

def model_math_symbol(numeberClass,img_height,img_width):
    num_classes = numeberClass
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        # layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),  # Look overhere!
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def model_handWriting_numbers(numeberClass):
    num_classes = numeberClass
    model = Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(15, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def normalize_image (img) :
    """
    Normalize image
    :param img: input image with range 0-255 int
    :return: normalized image (with range 0 - 1 float)
    """

    # taken from https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    info = np.iinfo(img.dtype)
    norm_img = img.astype(np.float64) / info.max

    # tensorflow mnist use black background - so change bg from white to black
    for i in range(len(norm_img)):
        for j in range(len(norm_img[i])):
            norm_img[i][j] = abs(norm_img[i][j] - 1)

    return norm_img

def create_model_old(numeberClass,img_height,img_width):

    ##old for mnist
    # data_augmentation = tf.keras.Sequential(
    #   [
    #     layers.experimental.preprocessing.RandomFlip("horizontal"),
    #     layers.experimental.preprocessing.RandomRotation(0.1),
    #     layers.experimental.preprocessing.RandomZoom(0.1),
    #   ]
    # )

    # model = Sequential([
    #   data_augmentation,
    #   # layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    #   layers.experimental.preprocessing.Rescaling(1./255),
    #   layers.Conv2D(16, 3, padding='same', activation='relu'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Conv2D(32, 3, padding='same', activation='relu'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Conv2D(64, 3, padding='same', activation='relu'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Dropout(0.2),  # Look overhere!
    #   layers.Flatten(),
    #   layers.Dense(128, activation=tf.nn.softmax)
    # ])

    # model = Sequential([
    #     layers.Flatten(),
    #     layers.Dense(128, activation=tf.nn.relu),
    #     layers.Dense(128, activation=tf.nn.relu),
    #     layers.Dense(128, activation=tf.nn.softmax)
    # ])


    num_classes = numeberClass
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        # layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),  # Look overhere!
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model