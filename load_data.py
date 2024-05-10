import os
import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))

images = []
labels = []

def read_path(path_name):
    for dir_item in os.listdir(path_name):
#windows修改过路径。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        full_path = f"{path_name}/{dir_item}"
        if os.path.isdir(full_path):  
            read_path(full_path)
        else:  
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(path_name)

    return images, labels

num_to_labels = {}
labels_to_num = {}
with open("./data/num_to_labels.txt", "r") as f:
    num_to_labels_list = [line.split(",") for line in f.readlines()]
    for i in num_to_labels_list:
        num_to_labels[i[0]] = int(i[1].strip())
        labels_to_num[int(i[1].strip())] = i[0]

def load_dataset(path_name):
    images, labels = read_path(path_name)
    images = np.array(images)
    labels = np.array([num_to_labels[label.split("/")[-1]] for label in labels])

    return images, labels


if __name__ == '__main__':
    images, labels = load_dataset("./data")
    print(labels)
