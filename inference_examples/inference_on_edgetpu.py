import argparse
import threading
from threading import Lock

from pycoral.utils import edgetpu
import numpy as np
import cv2

class Camera:

    last_frame = None
    last_ready = None
    lock = Lock()

    def __init__(self, rtsp_link):
        self.capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(self.capture,), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def is_open(self):
        return self.capture.isOpened()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()

    def read(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None


class InferenceModel:

    def __init__(self, model_file):
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_quant = self.interpreter.get_input_details()[0]['quantization']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]

        self.box_index = self.interpreter.get_output_details()[0]['index']
        self.box_quant = self.interpreter.get_output_details()[0]['quantization']
        self.score_index = self.interpreter.get_output_details()[2]['index']
        self.score_quant = self.interpreter.get_output_details()[2]['quantization']

    def process(self, img):
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        preprocessed = img
        # preprocessed = img.astype(np.float32)-127
        # preprocessed = (2.0 / 255) * img - 1.0
        # preprocessed = preprocessed/self.input_quant[0] + self.input_quant[1]
        # preprocessed = preprocessed.astype(np.int8)
        preprocessed = np.expand_dims(preprocessed, 0)
        self.interpreter.set_tensor(self.input_index, preprocessed)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.box_index)[0]
        scores = self.interpreter.get_tensor(self.score_index)[0]
        # boxes = (np.array(boxes, np.float32)-self.box_quant[1])*self.box_quant[0]
        # scores = (np.array(scores, np.float32)-self.score_quant[1])*self.score_quant[0]
        return boxes, scores


def camera_inference(model, video_id):
    cap = Camera(video_id)
    while True:
        img = cap.read()
        if img is None:
            continue
        img = cv2.resize(img, (1280, 720))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, scores = model.process(img)
        H = img.shape[0]
        W = img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box, score in zip(boxes, scores):
            if score > 0.5:
                box[0] = int(box[0] * H)
                box[1] = int(box[1] * W)
                box[2] = int(box[2] * H)
                box[3] = int(box[3] * W)
                box = box.astype(np.int32)
                img = cv2.rectangle(img, (box[1], box[0]), (box[3],box[2]), (0, 0, 255), 2)
        cv2.imshow('results', img)
        if cv2.waitKey(1) == ord('q'):
            break

def image_inference(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H = img.shape[0]
    W = img.shape[1]
    boxes, scores = model.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, score in zip(boxes, scores):
        if score > 0.4:
            box[0] = int(box[0] * H)
            box[1] = int(box[1] * W)
            box[2] = int(box[2] * H)
            box[3] = int(box[3] * W)
            box = box.astype(np.int32)
            img = cv2.rectangle(img, (box[1], box[0]), (box[3],box[2]), (0, 0, 255), 2)
    cv2.imshow('results', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite', type=str, help='tflite file path', required=True)
    parser.add_argument('--video', type=str, help='video file path or camera number')
    parser.add_argument('--image', type=str, help='image file path')
    args = parser.parse_args()
    model = InferenceModel(args.tflite)
    if args.video is not None:
        if args.video.isdigit():
            camera_inference(model, int(args.video))
        else:
            camera_inference(model, args.video)
    elif args.image is not None:
        image_inference(model, args.image)
    else:
        print('Nothing to do. Please check arguments.')
