import tensorflow as tf
import sys
import os


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 2d convolution of x and W, with stride 1 along every dimension and 0 padding
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max pooling, with a 2x2 kernel and a stride of 2 along x and y
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def define_model(model_input):
    with tf.variable_scope('layer_1'):
        # Weight and bias for the first convolutional layer
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        # convolution. Use of ReLU activation function
        h_conv1 = tf.nn.relu(conv2d(model_input, W_conv1) + b_conv1)
        # max pool
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('layer_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('fc_layer1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope('read_out_fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        predictions = tf.matmul(h_fc1, W_fc2) + b_fc2

    return predictions


def get_input_fn(file_pattern,
                 image_size=(28, 28),
                 shuffle=False,
                 batch_size=64,
                 num_epochs=None,
                 buffer_size=4096):

    def _img_string_to_tensor(image_string, image_size=(28, 28)):
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_decoded_as_float = tf.image.convert_image_dtype(
            image_decoded, dtype=tf.float32)  # (scale [0,1]
        # Resize to expected
        image_resized = tf.image.resize_images(
            image_decoded_as_float, size=image_size)

        return image_resized

    def _path_to_img(path):
        # Get the parent folder of this file to get its class
        label = tf.cond(
            tf.equal(tf.string_split([path], delimiter='/').values[-2], "dogs"),
            lambda: 0, lambda: 1)

        image_string = tf.read_file(path)  # read image and process it
        image_resized = _img_string_to_tensor(image_string, image_size)

        return image_resized, label

    def _input_fn():

        dataset = tf.data.Dataset.list_files(file_pattern)

        if shuffle:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(_path_to_img)
        dataset = dataset.batch(batch_size).prefetch(buffer_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return _input_fn


def model_fn(features, labels, mode, params):

    predictions = define_model(features)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=predictions))

    # Compute evaluation metrics.
    predicted_classes = tf.argmax(predictions, 1)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    data_directory = "/data/dogscats"
    model = tf.estimator.Estimator(model_fn, "./model_dir")

    train_files = os.path.join(data_directory, 'train', '**/*.jpg')
    train_input_fn = get_input_fn(train_files, shuffle=True, num_epochs=10)

    eval_files = os.path.join(data_directory, 'valid', '**/*.jpg')
    eval_input_fn = get_input_fn(eval_files, num_epochs=1)

    model.train(train_input_fn)
    model.evaluate(eval_input_fn)


if __name__ == "__main__":
    sys.exit(main())
