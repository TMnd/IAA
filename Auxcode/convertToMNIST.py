
#iterar cada imagem para MNIST e adcionar ao ficheiro csv.

import os
import csv
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import pathlib
from shutil import copyfile

def imageprepare(argv,imageName,folderName):
    # im = Image.open('../dados/hand_balance_croppedUsingOpenCV_MNISTStyle/+/tmp.png').convert('L')
    im = Image.open(cleanOpenCVRectColor(argv,imageName,folderName)).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # plt.imshow(newImage, interpolation='nearest')
    # plt.savefig('newImage.png')  # save MNIST image
    # plt.show()  # Show / plot that image

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    # print(tva)
    return tva

def cleanOpenCVRectColor(img,title,folderName):
    img = Image.open(img)

    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            compare = (12, 255, 36)
            a = img.getpixel((i, j))
            if a == compare:
                pixels[i, j] = (255, 255, 255)

    immg = ImageOps.grayscale(img).convert('L')

    output = f'../dados/hand_balance_croppedUsingOpenCV_MNISTStyle/{folderName}/{title}.png'

    # plt.imshow(gray_image)
    img.save(output)
    # plt.show()  # Show / plot that image

    return output

if __name__ == "__main__":
    # file = pathlib.Path("../dados/mnist_train.csv")
    file = pathlib.Path("../dados/mnist_train_WithSymbols_Balance_v2.csv")
    # if file.exists():
    #     #Clone csv file!
    #     fileSec = pathlib.Path("../dados/mnist_train_WithSymbols_Balance_2.csv")
    #     copyfile(file, fileSec)

    # for i in range(1):
    for folderName in os.listdir('../dados/hand_balance_croppedUsingOpenCV/'):
        if folderName == "14" or folderName == "15":
            print(folderName)
            for filename in os.listdir(f'../dados/hand_balance_croppedUsingOpenCV/{folderName}'):
                x = [imageprepare(f'../dados/hand_balance_croppedUsingOpenCV/{folderName}/{filename}',filename.split(".")[0],folderName)]  # file path here
                # print(len(x[0]))  # mnist IMAGES are 28x28=784 pixels
                # print(x[0])
                # Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
                newArr = [[0 for d in range(28)] for y in range(28)]
                k = 0
                for i in range(28):
                    for j in range(28):
                        newArr[i][j] = x[0][k]
                        k = k + 1

                # print(newArr)

                output = []
                output.append(folderName)

                for i in range(28):
                    for j in range(28):
                        output.append(newArr[i][j])

                # print(output)
                listToStr = ','.join([str(float(elem)) for elem in output])
                # print(listToStr)

                # with open(fileSec, 'a') as fd:
                with open(file, 'a') as fd:
                    fd.write(listToStr)
                    fd.write('\n')

                # plt.imshow(newArr)
                # plt.savefig('MNIST_IMAGE.png')  # save MNIST image
                # plt.show()  # Show / plot that image

                # os.remove('../dados/hand_balance_croppedUsingOpenCV_MNISTStyle/+/tmp.png')
    # else:
    #     print("File not exist")