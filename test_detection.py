import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import os
import shutil
from face_train import Model
import pandas as pd
import numpy as np
class FaceCaptureProcessor(VideoProcessorBase):
    def __init__(self, path_name, top_num, sign="frontalface") -> None:
        super().__init__()
        self.path_name = path_name
        self.top_num = top_num
        self.sign = sign
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.num_faces = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if self.num_faces < self.top_num:
                self.save_face(img, x, y, w, h)
                self.num_faces += 1
        return img

    def save_face(self, frame, x, y, w, h):
        img_name = f'{self.path_name}/{self.sign}_{self.num_faces}.jpg'
        face_image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
        cv2.imwrite(img_name, face_image)

def catch_face_info(iD, top_num):
    path_name = f"./data/{iD}"
    sign = "frontalface"
    faceCapProcess = FaceCaptureProcessor(path_name, top_num, sign)
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: faceCapProcess,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        st.write("处理中...")
        # st.warning("webrtc_ctx check!")
        # st.write(webrtc_ctx.video_processor)
    else:
        st.write("等待摄像头连接...")
    # st.warning("webrtc_ctx check property")
    # st.write(webrtc_ctx.video_transformer)
    # st.write(webrtc_ctx)
    
def catch_face_info_re(iD, top_num):
    path_name = iD
    sign = "frontalface"
    # faceCapProcess = FaceCaptureProcessor(path_name, top_num, sign)
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: FaceCaptureProcessor(path_name, top_num, sign),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        st.write("处理中...")
        # st.warning("webrtc_ctx check!")
        # st.write(webrtc_ctx.video_processor)
    else:
        st.write("等待摄像头连接...")
    # st.warning("webrtc_ctx check property")
    # st.write(webrtc_ctx.video_transformer)
    # st.write(webrtc_ctx)

def create_folder(directory, folder_name):
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 拼接文件夹路径
    folder_path = f"{directory}/{folder_name}"
    
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        st.success(f"文件夹 '{folder_name}' 已成功创建在 '{directory}' 下。")
    else:
        st.warning(f"文件夹 '{folder_name}' 在 '{directory}' 下已存在。")



def delete_folder(folder_path):
    try:
        # 递归删除文件夹及其所有内容
        shutil.rmtree(folder_path)
        st.success(f"文件夹 '{folder_path}' 及其内容已成功删除。")
    except OSError as e:
        st.warning(f"删除文件夹 '{folder_path}' 及其内容时出错: {e}")



def delete_images(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为图片格式
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)
                # 删除文件
                os.remove(file_path)




def face_test(model_path):
    img_name = f'./tmp/frontalface_19.jpg'
    
    # if st.button("签到"):
    # 加载图片
    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt2.xml"
    # 加载图片
    img = cv2.imread(img_name)
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    cascade = cv2.CascadeClassifier(cascade_path)

    # 利用分类器识别出哪个区域为人脸
    faceRect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    x, y, w, h = faceRect[0]
    # 截取脸部图像提交给模型识别这是谁
    image = img[y - 10: y + h + 10, x - 10: x + w + 10]
    # st.write(image)
    model = Model()
    # model.load_model("./model/test_model.keras")    # model 可能需要修改
    model.load_model(model_path)    # model 可能需要修改
    res = model.face_predict(image)
    labels_df = pd.read_csv('./data/num_to_labels_test.txt',sep=',',header=None)  # label 需要修改
    # st.write(labels_df)
    labels_df.columns = ['id','index']
    res = np.array(res)
    idx = np.argmax(res[0,:])
    # st.write("res is what:")
    # st.write(labels_df.loc[idx,'id'])
    return labels_df.loc[idx,'id']


def face_test2():
    catch_face_info_re('./tmp',20)
    
    if st.button("签到"):
        img_name = f'./tmp/frontalface_19.jpg'
        
        # if st.button("签到"):
        # 加载图片
        # 框住人脸的矩形边框颜色
        color = (0, 255, 0)
        # 人脸识别分类器本地存储路径
        cascade_path = "haarcascade_frontalface_alt2.xml"
        # 加载图片
        img = cv2.imread(img_name)
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 加载分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        x, y, w, h = faceRect[0]
        # 截取脸部图像提交给模型识别这是谁
        image = img[y - 10: y + h + 10, x - 10: x + w + 10]
        # st.write(image)
        model = Model()
        model.load_model("./model/test_model.keras")    # model 可能需要修改
        # model.load_model(model_path)    # model 可能需要修改
        res = model.face_predict(image)
        labels_df = pd.read_csv('./data/num_to_labels_test.txt',sep=',',header=None)  # label 需要修改
        # st.write(labels_df)
        labels_df.columns = ['id','index']
        res = np.array(res)
        idx = np.argmax(res[0,:])
        st.write("res is what:")
        st.write(labels_df.loc[idx,'id'])
        return labels_df.loc[idx,'id']


if __name__ == "__main__":
    face_test2()
