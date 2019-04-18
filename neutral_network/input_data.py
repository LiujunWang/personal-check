import xml.dom.minidom
from PIL import Image
import random
import tensorflow as tf
import os

trans_size = 64


def read_xmlfile(xml_file_path):
    """读取xml文件中的信息，提取出图片名和对应的personID"""
    # images_info = {}
    person_index = []
    classification = []

    try:
        dom = xml.dom.minidom.parse(xml_file_path)
        root = dom.documentElement
    except Exception as e:
        print("Error:没有找到文件或读取文件失败")
        raise e
    else:
        tag_list = root.getElementsByTagName("Item")
        for i in range(len(tag_list)):

            key = tag_list[i].getAttribute("imageName")
            value = tag_list[i].getAttribute("pedestrianID")
            '''
            key = tag_list[i].getAttribute("imageName")
            value = tag_list[i].getAttribute("pedestrianID")
            images_info[key] = value
            '''
            if not value in person_index:
                person_index.append(value)
                classification.append([key])

            else:
                index = person_index.index(value)
                classification[index].append(key)

        person_index = []

        # return images_info, person_index, classification
        # return images_info, person_index
        return classification


def create_record(xml_file_path, train_path):
    classification = read_xmlfile(xml_file_path)

    train_writer = tf.python_io.TFRecordWriter("train.tfrecords")

    test_writer = tf.python_io.TFRecordWriter("test.tfrecords")

    length_classification = len(classification)
    # random_list = range(0, length_classification)

    for i in range(length_classification):

        element_length = len(classification[i])
        # 制作训练集
        for j in range(element_length):

            image_path = train_path + "\\" + classification[i][j]
            image = Image.open(image_path)
            image = image.resize((trans_size, trans_size))
            image_fir = image.tobytes()
            # 正样本制作
            # for k in range(j + 1, element_length):
            k = (j + 1) % element_length
            image_path = train_path + "\\" + classification[i][k]
            image = Image.open(image_path)
            image = image.resize((trans_size, trans_size))
            image_sec = image.tobytes()
            label = 0
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'image_fir': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_fir])),
                    'image_sec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_sec])),

                }))
            train_writer.write(example.SerializeToString())
            # 制作负样本
            # random_index = random.sample(random_list,element_length)
            for k in range(2):
                while True:
                    random_index = random.randint(0, length_classification - 1)
                    if random_index != i:
                        break
                random_length = len(classification[random_index])
                temp = random.randint(0, random_length - 1)
                image_path = train_path + "\\" + classification[random_index][temp]
                image = Image.open(image_path)
                image = image.resize((trans_size, trans_size))
                image_sec = image.tobytes()
                label = 1
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'image_fir': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_fir])),
                        'image_sec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_sec])),

                    }))
                train_writer.write(example.SerializeToString())

        # 制作测试集正样本
        image_path = train_path + "\\" + classification[i][j - 1]
        image = Image.open(image_path)
        image = image.resize((trans_size, trans_size))
        image_fir = image.tobytes()
        # 正样本制作
        k = int((j - 1) / 2)
        image_path = train_path + "\\" + classification[i][k]
        image = Image.open(image_path)
        image = image.resize((trans_size, trans_size))
        image_sec = image.tobytes()
        label = 0
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image_fir': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_fir])),
                'image_sec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_sec])),

            }))
        test_writer.write(example.SerializeToString())
        while True:
            random_index = random.randint(0, length_classification - 1)
            if random_index != i:
                break
        random_length = len(classification[random_index])
        temp = random.randint(0, random_length - 1)
        image_path = train_path + "\\" + classification[random_index][temp]
        image = Image.open(image_path)
        image = image.resize((trans_size, trans_size))
        image_sec = image.tobytes()
        label = 1
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image_fir': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_fir])),
                'image_sec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_sec])),

            }))
        test_writer.write(example.SerializeToString())

    train_writer.close()
    test_writer.close()
    print("tfrecord文件生成成功")
    classification = []
    # random_list = []
    # random_index = []


def decode_file(filename_queue, batch_size=5, capacity=20):
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_fir': tf.FixedLenFeature([], tf.string),
            'image_sec': tf.FixedLenFeature([], tf.string),
        }
    )
    label = features['label']
    image_fir = features['image_fir']
    image_sec = features['image_sec']

    image_fir = tf.decode_raw(image_fir, tf.uint8)
    image_fir = tf.reshape(image_fir, [trans_size, trans_size, 3])
    image_sec = tf.decode_raw(image_sec, tf.uint8)
    image_sec = tf.reshape(image_sec, [trans_size, trans_size, 3])
    label = tf.cast(label, tf.int64)

    image_fir, image_sec, label = tf.train.batch([image_fir, image_sec, label],
                                                 batch_size=batch_size,
                                                 capacity=capacity,
                                                 num_threads=1)

    return image_fir, image_sec, label


if __name__ == "__main__":

    xml_file_path = "..\\doc\\train.xml"

    train_path = "..\\train"

    tfRecords_path = "train.tfrecords"
    test_tfRecords_path = "test.tfrecords"

    if not (os.path.isfile(tfRecords_path) and os.path.isfile(test_tfRecords_path)):
        print("tfrecords file开始生成...")
        create_record(xml_file_path, train_path)

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer(["train.tfrecords"], shuffle=False, num_epochs=5)

    image_fir, image_sec, label = decode_file(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for x in range(20):
            # image_fir_value, image_sec_value, label_value = sess.run([image_fir, image_sec, label])
            # print(image_fir)
            # print(image_sec)
            # print(label_value)
            print(label)

        coord.request_stop()
        coord.join(threads)
