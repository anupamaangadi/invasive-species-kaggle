import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from tools import TRAIN_PATH, create_paths, load_labels, LABELS_PATH, next_batch, load_all_images


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    # print(create_paths(TRAIN_PATH))
    invasive = load_labels(LABELS_PATH)
    # print(id)
    # print(invasive)
    image_paths = create_paths(TRAIN_PATH)

    sess = tf.Session()

    # build the model
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y_ = tf.placeholder(tf.float32, shape=[None])

    is_training = tf.placeholder(bool, name='phase')

    W_conv1 = weight_variable([3, 3, 3, 32])
    conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='SAME')
    bn1 = batch_norm(conv1, is_training=is_training)
    relu1 = tf.nn.relu(bn1)

    W_conv2 = weight_variable([3, 3, 32, 1])
    conv2_dw = tf.nn.depthwise_conv2d(relu1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bn2 = batch_norm(conv2_dw, is_training=is_training)
    relu2 = tf.nn.relu(bn2)

    W_conv3 = weight_variable([1, 1, 32, 64])
    conv3_pw = tf.nn.conv2d(relu2, W_conv3, [1, 1, 1, 1], padding='SAME')
    bn3 = batch_norm(conv3_pw, is_training=is_training)
    relu3 = tf.nn.relu(bn3)

    W_conv4 = weight_variable([3, 3, 64, 1])
    conv4_dw = tf.nn.depthwise_conv2d(relu3, W_conv4, strides=[1, 2, 2, 1], padding='SAME')
    bn4 = batch_norm(conv4_dw, is_training=is_training)
    relu4 = tf.nn.relu(bn4)

    W_conv5 = weight_variable([1, 1, 64, 128])
    conv5_pw = tf.nn.conv2d(relu4, W_conv5, [1, 1, 1, 1], padding='SAME')
    bn5 = batch_norm(conv5_pw, is_training=is_training)
    relu5 = tf.nn.relu(bn5)

    W_conv6 = weight_variable([3, 3, 128, 1])
    conv6_dw = tf.nn.depthwise_conv2d(relu5, W_conv6, strides=[1, 1, 1, 1], padding='SAME')
    bn6 = batch_norm(conv6_dw, is_training=is_training)
    relu6 = tf.nn.relu(bn6)

    W_conv7 = weight_variable([1, 1, 128, 128])
    conv7_pw = tf.nn.conv2d(relu6, W_conv7, [1, 1, 1, 1], padding='SAME')
    bn7 = batch_norm(conv7_pw, is_training=is_training)
    relu7 = tf.nn.relu(bn7)

    W_conv8 = weight_variable([3, 3, 128, 1])
    conv8_dw = tf.nn.depthwise_conv2d(relu7, W_conv8, strides=[1, 2, 2, 1], padding='SAME')
    bn8 = batch_norm(conv8_dw, is_training=is_training)
    relu8 = tf.nn.relu(bn8)

    W_conv9 = weight_variable([1, 1, 128, 256])
    conv9_pw = tf.nn.conv2d(relu8, W_conv9, [1, 1, 1, 1], padding='SAME')
    bn9 = batch_norm(conv9_pw, is_training=is_training)
    relu9 = tf.nn.relu(bn9)

    W_conv10 = weight_variable([3, 3, 256, 1])
    conv10_dw = tf.nn.depthwise_conv2d(relu9, W_conv10, strides=[1, 1, 1, 1], padding='SAME')
    bn10 = batch_norm(conv10_dw, is_training=is_training)
    relu10 = tf.nn.relu(bn10)

    W_conv11 = weight_variable([1, 1, 256, 256])
    conv11_pw = tf.nn.conv2d(relu10, W_conv11, [1, 1, 1, 1], padding='SAME')
    bn11 = batch_norm(conv11_pw, is_training=is_training)
    relu11 = tf.nn.relu(bn11)

    W_conv12 = weight_variable([3, 3, 256, 1])
    conv12_dw = tf.nn.depthwise_conv2d(relu11, W_conv12, strides=[1, 2, 2, 1], padding='SAME')
    bn12 = batch_norm(conv12_dw, is_training=is_training)
    relu12 = tf.nn.relu(bn12)

    W_conv13 = weight_variable([1, 1, 256, 512])
    conv13_pw = tf.nn.conv2d(relu12, W_conv13, [1, 1, 1, 1], padding='SAME')
    bn13 = batch_norm(conv13_pw, is_training=is_training)
    relu13 = tf.nn.relu(bn13)

    W_conv14 = weight_variable([3, 3, 512, 1])
    conv14_dw = tf.nn.depthwise_conv2d(relu13, W_conv14, strides=[1, 1, 1, 1], padding='SAME')
    bn14 = batch_norm(conv14_dw, is_training=is_training)
    relu14 = tf.nn.relu(bn14)

    W_conv15 = weight_variable([1, 1, 512, 512])
    conv15_pw = tf.nn.conv2d(relu14, W_conv15, [1, 1, 1, 1], padding='SAME')
    bn15 = batch_norm(conv15_pw, is_training=is_training)
    relu15 = tf.nn.relu(bn15)

    W_conv16 = weight_variable([3, 3, 512, 1])
    conv16_dw = tf.nn.depthwise_conv2d(relu15, W_conv16, strides=[1, 1, 1, 1], padding='SAME')
    bn16 = batch_norm(conv16_dw, is_training=is_training)
    relu16 = tf.nn.relu(bn16)

    W_conv17 = weight_variable([1, 1, 512, 512])
    conv17_pw = tf.nn.conv2d(relu16, W_conv17, [1, 1, 1, 1], padding='SAME')
    bn17 = batch_norm(conv17_pw, is_training=is_training)
    relu17 = tf.nn.relu(bn17)

    W_conv18 = weight_variable([3, 3, 512, 1])
    conv18_dw = tf.nn.depthwise_conv2d(relu17, W_conv18, strides=[1, 1, 1, 1], padding='SAME')
    bn18 = batch_norm(conv18_dw, is_training=is_training)
    relu18 = tf.nn.relu(bn18)

    W_conv19 = weight_variable([1, 1, 512, 512])
    conv19_pw = tf.nn.conv2d(relu18, W_conv19, [1, 1, 1, 1], padding='SAME')
    bn19 = batch_norm(conv19_pw, is_training=is_training)
    relu19 = tf.nn.relu(bn19)

    W_conv20 = weight_variable([3, 3, 512, 1])
    conv20_dw = tf.nn.depthwise_conv2d(relu19, W_conv20, strides=[1, 1, 1, 1], padding='SAME')
    bn20 = batch_norm(conv20_dw, is_training=is_training)
    relu20 = tf.nn.relu(bn20)

    W_conv21 = weight_variable([1, 1, 512, 512])
    conv21_pw = tf.nn.conv2d(relu20, W_conv21, [1, 1, 1, 1], padding='SAME')
    bn21 = batch_norm(conv21_pw, is_training=is_training)
    relu21 = tf.nn.relu(bn21)

    W_conv22 = weight_variable([3, 3, 512, 1])
    conv22_dw = tf.nn.depthwise_conv2d(relu21, W_conv22, strides=[1, 1, 1, 1], padding='SAME')
    bn22 = batch_norm(conv22_dw, is_training=is_training)
    relu22 = tf.nn.relu(bn22)

    W_conv23 = weight_variable([1, 1, 512, 512])
    conv23_pw = tf.nn.conv2d(relu22, W_conv23, [1, 1, 1, 1], padding='SAME')
    bn23 = batch_norm(conv23_pw, is_training=is_training)
    relu23 = tf.nn.relu(bn23)

    W_conv24 = weight_variable([3, 3, 512, 1])
    conv24_dw = tf.nn.depthwise_conv2d(relu23, W_conv24, strides=[1, 2, 2, 1], padding='SAME')
    bn24 = batch_norm(conv24_dw, is_training=is_training)
    relu24 = tf.nn.relu(bn24)

    W_conv25 = weight_variable([1, 1, 512, 1024])
    conv25_pw = tf.nn.conv2d(relu24, W_conv25, [1, 1, 1, 1], padding='SAME')
    bn25 = batch_norm(conv25_pw, is_training=is_training)
    relu25 = tf.nn.relu(bn25)

    W_conv26 = weight_variable([3, 3, 1024, 1])
    conv26_dw = tf.nn.depthwise_conv2d(relu25, W_conv26, strides=[1, 1, 1, 1], padding='SAME')
    bn26 = batch_norm(conv26_dw, is_training=is_training)
    relu26 = tf.nn.relu(bn26)

    W_conv27 = weight_variable([1, 1, 1024, 1024])
    conv27_pw = tf.nn.conv2d(relu26, W_conv27, [1, 1, 1, 1], padding='SAME')
    bn27 = batch_norm(conv27_pw, is_training=is_training)
    relu27 = tf.nn.relu(bn27)

    # W_pool28 = weight_variable([1, 7, 7, 1])
    avg_pool28 = tf.nn.avg_pool(relu27, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    bn28 = batch_norm(avg_pool28, is_training=is_training)
    relu28 = tf.nn.relu(bn28)

    fc29 = tf.reshape(relu28, shape=[-1, 1024])
    W_fc29 = weight_variable([1024, 1])

    y_conv = tf.reshape(tf.matmul(fc29, W_fc29), shape=[-1])

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # loss = tf.reduce_mean(tf.cast(cross_entropy, tf.float32))
    accuracy = tf.reduce_mean(tf.abs(tf.subtract(y_conv, y_)))

    sess.run(tf.global_variables_initializer())
    all_images = load_all_images(image_paths)
    labels_raw = np.asarray(list(invasive.values()))

    train_images = all_images[:2250]
    train_labels = labels_raw[:2250]
    valid_images = all_images[2250:]
    valid_labels = labels_raw[2250:]

    X_valid, y_valid = next_batch(valid_images, labels=valid_labels, size=len(valid_images))

    for i in range(20000):
        X_train, y_train = next_batch(train_images, grayscale=False, size=20, labels=train_labels)
        sess.run(train_step, feed_dict={x: X_train, y_: y_train, is_training: True})
        if i % 100 == 0:
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                           y_: y_valid,
                                                           is_training: False})
            print(f'Step {i}, valid accuracy: {100 - valid_accuracy * 100}%')
            # print(f'Test accuracy {accuracy.eval(feed_dict={x: X, y:})}')

            # plt.imshow('gray_image', X[0]*255.)
            # print(y[0])
            # cv2.imshow('gray_image', X[0])
            # cv2.waitKey(0)
            # plt.show()
