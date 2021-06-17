import argparse

from train.input_pipeline import make_tfdataset

import tensorflow as tf

def convert_tflite_int8(saved_model_path, calb_data, output_name, quant_level=0):
    """
    quant_level == 0:
        weights only quantzation, no requires calibration data.
    quant_level == 1:
        Full quantization for supported operators.
        It remains float for not supported operators.
    quant_level == 2:
        Full quantization for all operators.
        It can not be converted if the model contains not supported operators.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        for n, data in enumerate(calb_data.take(1000)):
            x = data[0]
            if n % 10 == 0:
                print(n, 'images are processed.')
            # Get sample input data as a numpy array in a method of your choosing.
            # The batch size should be 1.
            # So the shape of the x should be (1, height, width, channel)
            yield [x]
    if quant_level == 1:
        converter.representative_dataset = representative_dataset_gen
    elif quant_level == 2:
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
    tflite_quant_model = converter.convert()
    with open(output_name, 'wb') as f:
        f.write(tflite_quant_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, required=True,
        help='trained saved model path')
    parser.add_argument('--dataset', type=str, required=True,
        help='calibration dataset (tfrecord)')
    parser.add_argument('--image_size', type=str, required=True,
        help='image height and width. ex) 112,112')
    parser.add_argument('--quant_level', type=int, required=False,
        default=0, help='quantization level 0 ~ 2')
    parser.add_argument('--output', type=str, required=True,
        help='output file name')
    args = parser.parse_args()
    img_size = args.image_size.split(',')
    width = int(img_size[0])
    height = int(img_size[1])
    quant_level = args.quant_level
    dataset = make_tfdataset(args.dataset, 1, (width, height))
    convert_tflite_int8(args.saved_model, dataset, args.output, quant_level)
