##
# TEMA 2 - MNIST hand written digit database
# Permite classificar imagens ou escrever os numeros em um canvas ao usar o modelo treinado no ficheiro
# tensorflow_MNIST_Handwriteen_DL_Model_Training.py
##

import cv2
import pathlib
import PIL
import os
import tensorflow as tf
from externalFunctions import model_handWriting_numbers, normalize_image, model_math_symbol
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import ImageGrab
import os, random

mathSymbol = {10: '+', 11: ' /', 12: '*', 13: '-', 14: '(', 15: ')'}

def checkNumber(img):
    img = PIL.Image.open(pathlib.Path(img)).convert('L').resize((28, 28), PIL.Image.ANTIALIAS)
    img = np.array(img)
    img = normalize_image(img)
    plt.imshow(img, cmap="gray")
    plt.show()

    my_tests = np.array([img])
    predictions = numberWriteHandingModel.predict([my_tests])
    score = tf.nn.softmax(predictions[0])

    disc = {}
    for i in range(len(predictions[0])):
        disc[predictions[0][i]] = i

    prediction = np.argmax(predictions[0])

    output = prediction if prediction not in mathSymbol else mathSymbol[prediction]

    print(score)
    outputMsg = "checkNumber - This image most likely belongs to {}.".format(output)
    print(outputMsg)

    return str(output)

def imageResult(result):
    randomImage = random.choice(os.listdir(f'dados/hand_balance/{result}'))  # change dir name to whatever
    resultImage = cv2.imread(f'dados/hand_balance/{result}/{randomImage}')
    cv2.imshow('result', resultImage)

#Canvas controls
def xy(event):
    "Takes the coordinates of the mouse when you click the mouse"
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    """Creates a line when you drag the mouse
    from the point where you clicked the mouse to where the mouse is now"""
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), width=5)
    # this makes the new starting point of the drawing
    lastx, lasty = event.x, event.y

def clean(event):
    canvas.delete("all")

def quit(event):
    canvas.quit()

def calc(event):
    #Save file
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    im = ImageGrab.grab((x, y, x1, y1))
    captureUrl = "ficheirosParaTestar/captured.png"
    im.save(captureUrl)

    #Calc
    image = cv2.imread(captureUrl)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    #Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boundary = []
    for c, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        boundary.append((x, y, w, h))
    count = np.asarray(boundary)
    max_width = np.sum(count[::, (0, 2)], axis=1).max()
    max_height = np.max(count[::, 3])
    nearest = max_height * 1.4
    ind_list = np.lexsort((count[:, 0], count[:, 1]))

    c = count[ind_list]
    sorted_ctrs = sorted(c, key=lambda x: x[0])

    ROI_number = 0
    equation = ""

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = ctr[0], ctr[1], ctr[2], ctr[3]

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        cv2.imshow('segment no:' + str(i), roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        cv2.waitKey(0)
        file = f'dados/tobedeleted/ROI_{ROI_number}.png'
        cv2.imwrite(file, roi)
        equation = equation+checkNumber(file)
        ROI_number += 1
        cv2.destroyAllWindows()

    #calc
    print(equation)
    result = eval(equation)
    print(result)

    if 10 > int(result) >= 0:
        imageResult(result)

    cv2.imshow('image', image)
    cv2.waitKey()

    os.remove(captureUrl)

if __name__ == "__main__":

    # Carregar o model MNIST
    numberWriteHandingModel = model_handWriting_numbers(16)
    checkpoint_path = "savedModels/handWriting_WithPlusSymbol_balance2_2_With()_100/cp.ckpt.index"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    numberWriteHandingModel.load_weights(latest)

    mode = "s"

    if mode == "s":
        image = cv2.imread('ficheirosParaTestar/30_2_d_2-1.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

        #Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        boundary = []
        for c, cnt in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(cnt)
            boundary.append((x, y, w, h))
        count = np.asarray(boundary)
        max_width = np.sum(count[::, (0, 2)], axis=1).max()
        max_height = np.max(count[::, 3])
        nearest = max_height * 1.4
        ind_list = np.lexsort((count[:, 0], count[:, 1]))

        c = count[ind_list]
        sorted_ctrs = sorted(c, key=lambda x: x[0])

        ROI_number = 0
        calc = ""

        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = ctr[0], ctr[1], ctr[2], ctr[3]

            # Getting ROI
            roi = image[y:y + h, x:x + w]

            # show ROI
            cv2.imshow('segment no:' + str(i), roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
            cv2.waitKey(0)
            file = f'dados/tobedeleted/ROI_{ROI_number}.png'
            cv2.imwrite(file, roi)
            calc = calc+checkNumber(file)
            ROI_number += 1
            cv2.destroyAllWindows()

        #calc
        print(calc)
        print(eval(calc))

        cv2.imshow('image', image)
        cv2.waitKey()
    else:
        # Canvas
        root = tk.Tk()
        root.geometry("1600x700")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        canvas = tk.Canvas(root)
        canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        canvas.bind("<Button-1>", xy)
        canvas.bind("<Button-3>", calc)
        canvas.bind("<B1-Motion>", addLine)
        root.bind("<Control-z>", clean)
        root.bind("<Control-q>", quit)

        root.mainloop()



    #Static
    # # Load image, grayscale, Gaussian blur, adaptive threshold
    #