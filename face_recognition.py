"""
通过训练生成的keras模型，进行人脸识别
"""
import pickle

import cv2
import numpy as np
from face_train import Model


def Video_reg(model_path, class_path, Video=None):
    labels_to_num = {}
    with open(f"./{class_path}/num_to_labels.txt", "r") as f:
        num_to_labels_list = [line.split(",") for line in f.readlines()]
        for i in num_to_labels_list:
            labels_to_num[int(i[1].strip())] = i[0]

    # 加载模型
    model = Model()
    model.load_model(file_path=model_path)

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt2.xml"

    if Video is None:
        # 捕获指定摄像头的实时视频流
        cap = cv2.VideoCapture(0)
        # 循环检测识别人脸
        while True:
            ret, frame = cap.read()  # 读取一帧视频
            if ret is True:

                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)

            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    face_res = model.face_predict(image)
                    faceID = np.argsort(face_res[0])[-1]
                    if face_res[0][faceID] >= 0.8:  # 设置阈值大于该概率才输出
                        face_num = labels_to_num[int(faceID)]  # 根据标签拿到对应学号

                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        # 文字提示是谁
                        cv2.putText(frame, f'{face_num}:{face_res[0][faceID] * 100:.1f}%',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽
                    else:
                        pass

            cv2.imshow("识别窗口", frame)

            # 等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(Video)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        # 循环读取视频的每一帧
        while True:
            # 读取下一帧
            ret, frame = cap.read()
            if ret is True:
                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)

            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    face_res = model.face_predict(image)
                    faceID = np.argsort(face_res[0])[-1]
                    if face_res[0][faceID] >= 0.8:  # 设置阈值大于该概率才输出
                        face_num = labels_to_num[int(faceID)]  # 根据标签拿到对应学号

                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        # 文字提示是谁
                        cv2.putText(frame, f'{face_num}:{face_res[0][faceID] * 100:.1f}%',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽
                    else:
                        pass

            cv2.imshow("识别窗口", frame)

            # 等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cv2.destroyAllWindows()


def img_reg(model_path, class_path, img_path, resize_k=1.0):
    labels_to_num = {}
    with open(f"./{class_path}/num_to_labels.txt", "r") as f:
        num_to_labels_list = [line.split(",") for line in f.readlines()]
        for i in num_to_labels_list:
            labels_to_num[int(i[1].strip())] = i[0]

    # 加载模型
    model = Model()
    model.load_model(file_path=model_path)

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt2.xml"

    # 加载图片
    img = cv2.imread(img_path)
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    cascade = cv2.CascadeClassifier(cascade_path)

    # 利用分类器识别出哪个区域为人脸
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            # 截取脸部图像提交给模型识别这是谁
            image = img[y - 10: y + h + 10, x - 10: x + w + 10]
            face_res = model.face_predict(image)
            faceID = np.argsort(face_res[0])[-1]
            cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
            if face_res[0][faceID] >= 0.8:  # 设置阈值大于该概率才输出
                face_num = labels_to_num[int(faceID)]  # 根据标签拿到对应学号

                # cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                # 文字提示是谁
                cv2.putText(img, f'{face_num}:{face_res[0][faceID] * 100:.1f}%',
                            (x + 30, y + 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
        img = cv2.resize(img, (0, 0), fx=resize_k, fy=resize_k)
        cv2.imshow("识别窗口", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Video_reg(model_path='./model/face_model_class_05.keras', class_path='data_class', Video='class.mp4')
    # img_reg(model_path='./model/face_model_class_04.keras', class_path='data_class',
    #         img_path="class02.jpg", resize_k=0.3)


#目前recognition程序是那个同学传给我的，需要改的是将此识别程序改为用streamlit自带的实现，只用实现导入照片识别、
#然后将识别到的同学的学号给我，照片识别的名字用日期命名，同时发给我识别日期
#123.249.37.195