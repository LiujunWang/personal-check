import xml.dom.minidom
#import numpy as np
from PIL import Image
import tensorflow as tf
#import os

trans_size = 64

def read_xmlfile(xml_file_path):
    """读取xml文件中的信息，提取出图片名和对应的personID并初始化分类"""
    images_info = {}
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
            images_info[key] = value
            if not value in classification:
                classification.append(value)
        
        return images_info, classification

def create_record(xml_file_path, train_path):
    """制作测试集和训练集的tfRecords文件"""
    images_info, classification = read_xmlfile(xml_file_path)

    TOTAL_SIZE = len(images_info)
    count = 0

    train_writer = tf.python_io.TFRecordWriter("train.tfrecords")
    test_writer = tf.python_io.TFRecordWriter("test.tfrecords")

    for key, value in images_info.items():

        image_path = train_path +"\\"+ key
        image = Image.open(image_path)
        image = image.resize((trans_size, trans_size))
        image_raw = image.tobytes()
        label = classification.index(value)
        
        example = tf.train.Example(
                features = tf.train.Features(feature = {
                    'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
                    'image_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_raw])),
                }))

        if count <= 0.9 * TOTAL_SIZE:
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

        count = count + 1
        
    
    train_writer.close()
    test_writer.close()

    print("tfrecord文件生成成功")



def decode_file(filename_queue, batch_size = 5, capacity = 20):
    
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    label = features['label']
    image = features['image_raw']
    
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [trans_size, trans_size, 3])
    label = tf.cast(label, tf.int64)

    image, label = tf.train.batch([image, label], batch_size = batch_size,                                          capacity = capacity,
                                                  num_threads = 1)
    

    return image, label

if __name__ == "__main__":


    xml_file_path = "..\\doc\\train.xml"
    #train_path = "..\\train"

    #create_record(xml_file_path, train_path)
    images_info, classification = read_xmlfile(xml_file_path)
    print(classification[35])
    '''

    xml_file_path = "..\\doc\\train.xml"

    

    tfRecords_path = "train.tfrecords"
    test_tfRecords_path = "test.tfrecords"

    if not (os.path.isfile(tfRecords_path) and os.path.isfile(test_tfRecords_path)):
        print("tfrecords file开始生成...")
        create_record(xml_file_path, train_path)

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer(["train.tfrecords"], shuffle = False, num_epochs = 5)

    image, label = decode_file(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for x in range(20):
            
            image_value, label_value = sess.run([image, label])
            print(image_value)
        
            print(label_value)

        coord.request_stop()
        coord.join(threads)
        '''

    
