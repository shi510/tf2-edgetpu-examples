import argparse

import cv2
import numpy as np
import tensorflow as tf


class InferenceModel:

    def __init__(self, model_file):
        self.interpreter = tf.lite.Interpreter(model_file)
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]
        self.input_dtype = self.interpreter.get_input_details()[0]['dtype']

        self.box_index = self.interpreter.get_output_details()[0]['index']
        self.label_index = self.interpreter.get_output_details()[1]['index']
        self.score_index = self.interpreter.get_output_details()[2]['index']

    def process(self, img):
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        if self.input_dtype == np.float32:
            img = (img / 255.0).astype(np.float32)
        img = np.expand_dims(img, 0)
        self.interpreter.set_tensor(self.input_index, img)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.box_index)[0]
        scores = self.interpreter.get_tensor(self.score_index)[0]
        labels = self.interpreter.get_tensor(self.label_index)[0]
        return boxes, scores, labels


def video_inference(model, video_id, out_path):
    out_size = (1280, 720)
    cap = cv2.VideoCapture(video_id)
    writer = cv2.VideoWriter(filename=out_path,
        fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        frameSize=out_size)
    num_frams = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frams):
        _, img = cap.read()
        if img is None:
            continue
        img = cv2.resize(img, out_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = (img / 255.0).astype(np.float32)
        boxes, scores = model.process(preprocessed)
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
        writer.write(img)
        print('{} / {} frames are processed.'.format(i, num_frams))

        # cv2.imshow('results', img)
        # if cv2.waitKey(1) == ord('q'):
        #     break

def image_inference(model, img_path, out_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H = img.shape[0]
    W = img.shape[1]
    preprocessed = img
    boxes, scores, labels = model.process(preprocessed)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.4:
            box[0] = int(box[0] * H)
            box[1] = int(box[1] * W)
            box[2] = int(box[2] * H)
            box[3] = int(box[3] * W)
            box = box.astype(np.int32)
            img = cv2.rectangle(img, (box[1], box[0]), (box[3],box[2]), (0, 0, 255), 2)
            print(label, score)
    cv2.imwrite(out_path, img)
    print('Done, saved to {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite', type=str, help='tflite file path', required=True)
    parser.add_argument('--video', type=str, help='video file path')
    parser.add_argument('--image', type=str, help='image file path')
    parser.add_argument('--output', type=str, help='output file path', required=True)
    args = parser.parse_args()
    model = InferenceModel(args.tflite)
    if args.video is not None:
        video_inference(model, args.video, args.output)
    elif args.image is not None:
        image_inference(model, args.image, args.output)
    else:
        print('Nothing to do. Please check arguments.')
