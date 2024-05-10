import cv2
import numpy as np
import streamlit as st
from face_train import Model
from load_data import labels_to_num
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class FaceRecognizer(VideoProcessorBase):
    def __init__(self, model_path, cascade_path):
        super().__init__()
        self.model = Model()
        self.model.load_model(file_path=model_path)
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def process(self, frame):
        image = frame.to_ndarray(format="bgr24")
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        for (x, y, w, h) in face_rects:
            face_image = frame_gray[y - 10: y + h + 10, x - 10: x + w + 10]
            face_res = self.model.face_predict(face_image)
            faceID = np.argsort(face_res[0])[-1]

            if face_res[0][faceID] >= 0.8:
                face_num = labels_to_num[faceID]
                cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), thickness=2)
                cv2.putText(image, f'{face_num}:{face_res[0][faceID]*100}%', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                st.success(f'已成功识别人脸，学号：{face_num}')

        return image

def main():
    st.title("人脸识别")

    st.markdown("### 视频捕获")
    st.markdown("请在下方查看摄像头捕获的视频流，并进行人脸识别。")

    model_path = "./model/face_model_03.keras"
    cascade_path = "haarcascade_frontalface_alt2.xml"

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: FaceRecognizer(model_path, cascade_path),
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        st.write("识别中...")
    else:
        st.write("等待摄像头连接...")

if __name__ == '__main__':
    main()
