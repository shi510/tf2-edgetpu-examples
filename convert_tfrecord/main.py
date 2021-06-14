import argparse
import os
import json

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def make_tfrecord(root_path, out_file, example_json):
    tf_file = tf.io.TFRecordWriter(out_file)
    count = 0
    for n, img_name in enumerate(example_json):
        data = example_json[img_name]['detection_label']
        with open(os.path.join(root_path, img_name), 'rb') as jpeg_file:
            jpeg_bytes = jpeg_file.read()
        if jpeg_bytes is None:
            print('{} is skipped because it cannot read the file.'.format(img_name))
            continue
        num_ids = len(data['class_ids'])
        for i in range(num_ids):
            box = data['box_list'][i]
            feature = {
                'jpeg': _bytes_feature(jpeg_bytes),
                'label': _int64_feature(data['class_ids'][i]),
                'x1': _float_feature(box[0]),
                'y1': _float_feature(box[1]),
                'x2': _float_feature(box[2]),
                'y2': _float_feature(box[3])
            }
            exam = tf.train.Example(features=tf.train.Features(feature=feature))
            tf_file.write(exam.SerializeToString())
        count = count + 1
        if (n+1) % 1000 == 0:
            print('{} images saved.'.format(n+1))
    tf_file.close()
    print('generating tfrecord is finished.')
    print('total number of images: {}'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
        help='absolute path of images in json_file')
    parser.add_argument('--json_file', type=str, required=True,
        help='examples including image relative path, label and bounding box')
    parser.add_argument('--output', type=str, required=True,
        help='tfrecord file name including extension')
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.loads(f.read())
    make_tfrecord(args.root_path, args.output, data)
