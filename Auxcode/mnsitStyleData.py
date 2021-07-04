import cv2
import pathlib
import PIL
import os
import tensorflow as tf
from externalFunctions import model_handWriting_numbers, normalize_image, model_math_symbol
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    numberWriteHandingModel = model_handWriting_numbers(10)
    checkpoint_path = "../savedModels/handWriting_v3_100/cp.ckpt.index"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    numberWriteHandingModel.load_weights(latest)

    file = "../dados/hand_balance_croppedUsingOpenCV/10/_+_0.png"
    img = PIL.Image.open(pathlib.Path(file)).convert('L').resize((100, 100), PIL.Image.ANTIALIAS)
    img = np.array(img)
    img = normalize_image(img)
    plt.imshow(img, cmap="gray")
    plt.show()

    my_tests = np.array([img])
    predictions = numberWriteHandingModel.predict([my_tests])
    score = tf.nn.softmax(predictions[0])

    print(predictions[0])

    # print("Prediction value:", np.argmax(predictions[0]))
    # print("Percentage value", np.max(score) * 100)
    #
    # print(np.argmax(predictions[0]))

    print("---------------")

    file = "../ficheirosParaTestar/4.png"
    img = PIL.Image.open(pathlib.Path(file)).convert('L').resize((28, 28), PIL.Image.ANTIALIAS)
    img = np.array(img)
    img = normalize_image(img)
    plt.imshow(img, cmap="gray")
    plt.show()

    my_tests = np.array([img])
    predictions = numberWriteHandingModel.predict([my_tests])
    score = tf.nn.softmax(predictions[0])

    print(predictions[0])

    # print("Prediction value:", np.argmax(predictions[0]))
    # print("Percentage value", np.max(score) * 100)
    #
    # print(np.argmax(predictions[0]))