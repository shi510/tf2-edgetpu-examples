import tensorflow as tf
import tensorflow_addons as tfa


TF_AUTOTUNE = tf.data.AUTOTUNE


def random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)

def gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

def random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.2)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x

def blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)


    def mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)


    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))

def make_tfdataset(tfrecord_file_path, batch_size, img_shape):
    ds = tf.data.TFRecordDataset(tfrecord_file_path)

    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.float32),
            'y1': tf.io.FixedLenFeature((), tf.float32),
            'x2': tf.io.FixedLenFeature((), tf.float32),
            'y2': tf.io.FixedLenFeature((), tf.float32)
        }
        example = tf.io.parse_single_example(serialized, description)
        # image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        image = tf.io.decode_image(example['jpeg'], channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        label = example['label']
        box = [ example['y1'], example['x1'], example['y2'], example['x2'] ]
        image = tf.image.resize(image, img_shape)
        return image, label, box

    def map_eager_decorator(func):
        def wrapper(images, labels, boxes):
            return tf.py_function(
                func,
                inp=(images, labels, boxes),
                Tout=(images.dtype, labels.dtype, boxes.dtype)
            )
        return wrapper

    def _preprocess(imgs, labels, boxes):
        imgs = (2.0 / 255.0) * imgs - 1.0
        labels = tf.one_hot(tf.cast(labels, tf.int32), 1)
        labels = tf.cast(labels, tf.float32)
        labels = [tf.expand_dims(i, 0) for i in labels]
        boxes = [tf.expand_dims(i, 0) for i in boxes]
        return imgs, labels, boxes

    def _preprocess_images(imgs, labels, boxes):
        imgs = (2.0 / 255.0) * imgs - 1.0
        return imgs, labels, boxes

    ds = ds.map(_read_tfrecord)
    # ds = ds.shuffle(5000)
    ds = ds.batch(batch_size)
    # ds = ds.map(map_eager_decorator(_preprocess))
    ds = ds.map(_preprocess_images)
    augmentations = [random_color, gray, blur]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        ds = ds.map(lambda img, label, box: (tf.cond(choice > 0.5, lambda: f(img), lambda: img), label, box),
            num_parallel_calls=TF_AUTOTUNE)
    # ds = ds.map(lambda img, label, box: (tf.clip_by_value(img, 0., 1.), label, box), num_parallel_calls=TF_AUTOTUNE)
    ds = ds.prefetch(TF_AUTOTUNE)

    return ds
