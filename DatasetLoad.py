import numpy as np
import gzip
import os
import platform
import pickle
import torch
from mpmath.identification import transforms
from torchvision.datasets import ImageFolder
import cv2
from WHDY_vanilla_malicious_involved_fedavg.getData import GetDataSet


class DatasetLoad(object):
    def __init__(self, dataSetName, isIID):
        self.img_size = None
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        if self.name == 'feyan':
            self.pneumonia_dataset_construct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        data_dir = 'data/meMNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 'test-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 'test-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)
        # CPU reduce size
        # train_images = train_images[:60]
        # train_labels = train_labels[:60]
        # test_images = test_images[:60]
        # test_labels = test_labels[:60]
        print(train_labels.shape)
        print(train_images.shape)
        # 60000 data points
        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]



        self.test_data = test_images
        self.test_label = test_labels

    def pneumonia_dataset_construct(self, isIID):
        # 定义图像预处理
        data_dir = 'data789'
        test_images_path  = 'chest_xray/test/image'
        test_labels_path  = 'chest_xray/test/label'
        train_images_path = 'chest_xray/train/image'
        train_labels_path = 'chest_xray/train/label'
        train_images = extract_imagesf(train_images_path)
        train_labels = extract_labelsf(train_labels_path, num_classes=2)
        test_images = extract_imagesf(test_images_path)
        test_labels = extract_labelsf(test_labels_path, num_classes=2)
        # CPU reduce size
        # train_images = train_images[:60]
        # train_labels = train_labels[:60]
        # test_images = test_images[:60]
        # test_labels = test_labels[:60]
        print(train_labels.shape)
        print("看看我的标签有没有问题")
        print(train_images.shape)
        # 60000 data points
        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


import os
from PIL import Image
import numpy as np
import tensorflow as tf

def extract_imagesf(path: str) -> np.ndarray:
    """从指定路径加载并预处理图像数据"""
    images = []

    # 遍历路径下的所有文件
    for filename in sorted(os.listdir(path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 构建完整文件路径
            img_path = os.path.join(path, filename)

            try:
                # 打开并转换为灰度图
                img = Image.open(img_path).convert('L')
                # 调整图像大小为224x224（可根据需要修改）
                img = img.resize((28, 28))
                # 转换为numpy数组
                img_array = np.array(img)
                # 添加通道维度 (H, W) -> (H, W, 1)
                img_array = np.expand_dims(img_array, axis=-1)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # 转换为numpy数组并返回
    return np.array(images)


def extract_labelsf(label_dir: str, num_classes: int) -> np.ndarray:
    """从指定路径加载并处理标签数据，并将其转换为独热编码形式"""
    labels = []
    # 历标签目录中的所有文件
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            # 读取标签文件的内容
            with open(file_path, 'r') as file:
                label = file.read().strip()  # 读取文件内容并去除两端空白字符
                # 将标签转换为整数并添加到列表中
                labels.append(int(label))
    # 将列表转换为NumPy数组
    labels = np.array(labels)
    print(labels[labels >= num_classes])
    # 将标签转换为独热编码形式
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels




if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

# add Gussian Noise to dataset
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
