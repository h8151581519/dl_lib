import gzip
import numpy as np

def mnist_imgread(path, is_flatten=False):
    source_file = gzip.open(path, 'r')
    # magic num 2051
    magic_num = np.frombuffer(source_file.read(4), dtype=np.uint8)
    magic_num = magic_num[2] * 256 + magic_num[3]
    # num of img 60000
    num_of_img = np.frombuffer(source_file.read(4), dtype=np.uint8)
    num_of_img = num_of_img[2] * 256 + num_of_img[3]
    # img weight
    img_w = np.frombuffer(source_file.read(4), dtype=np.uint8)
    img_w = img_w[2] * 256 + img_w[3]
    # img height
    img_h = np.frombuffer(source_file.read(4), dtype=np.uint8)
    img_h = img_h[2] * 256 + img_h[3]

    # read images
    buf = source_file.read(img_w * img_h * num_of_img)
    training_images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    training_images = training_images.reshape((num_of_img, img_w * img_h, 1)) if is_flatten else training_images.reshape((num_of_img, img_w, img_h, 1))
    return training_images / 255

def mnist_labread(path):
    source_file = gzip.open(path, 'r')
    # magic num 2049
    magic_num = np.frombuffer(source_file.read(4), dtype=np.uint8)
    magic_num = magic_num[2] * 256 + magic_num[3]
    # num of lab 10000
    num_of_lab = np.frombuffer(source_file.read(4), dtype=np.uint8)
    num_of_lab = num_of_lab[2] * 256 + num_of_lab[3]

    # read labels
    buf = source_file.read(num_of_lab)
    testing_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    testing_labels = testing_labels.reshape(num_of_lab, 1)
    return testing_labels