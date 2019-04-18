import numpy as np
import tensorflow as tf
import os
from PIL import Image
import operator

trans_size = 64

def file_name(file_dir):

    for root, dirs, files in os.walk(file_dir):

        return files

def image_process(image_path):

    image = Image.open(image_path)
    image = image.resize((trans_size, trans_size))
    image = np.array(image).reshape((1, trans_size, trans_size, 3))
    
    return image

def query(query_image_path,reference_path):

    reference_path_list = []
    result_dic = {}
    result = {}

    sess = tf.Session()
    saver = tf.train.import_meta_graph('../Model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('../Model/'))

    graph = tf.get_default_graph()
    placeholder_fir = graph.get_tensor_by_name("placeholder_fir:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    
    input_image_fir = image_process(query_image_path)
    op_to_restore = graph.get_tensor_by_name("InceptionV3/SpatialSqueeze:0")
    image_fir_value = sess.run(op_to_restore,feed_dict = {placeholder_fir:input_image_fir, keep_prob:1})
    '''
    input_image_sec = image_process(reference_path)
    op_to_restore = graph.get_tensor_by_name("InceptionV3/SpatialSqueeze:0")
    image_sec_value = sess.run(op_to_restore,feed_dict = {placeholder_fir:input_image_sec, keep_prob:1})

    distance = np.sqrt(np.sum(np.square(image_fir_value - image_sec_value)))
    print(distance)
    '''
    reference_path_list = file_name(reference_path)
    #print(len(reference_path_list))
    
    for path_name in reference_path_list:


        reference_image_path = reference_path + "/" + path_name

        input_image_sec = image_process(reference_image_path)

        op_to_restore = graph.get_tensor_by_name("InceptionV3/SpatialSqueeze:0")
        image_sec_value = sess.run(op_to_restore,feed_dict = {placeholder_fir:input_image_sec, keep_prob:1})

        distance = np.sqrt(np.sum(np.square(image_fir_value - image_sec_value)))
        if distance <= 1:
            result_dic[path_name] = distance
    print(result_dic)
    result_dic = sorted(result_dic.items(),key = operator.itemgetter(1))
    for x in range(len(result_dic)):
        key, value = result_dic[x]
        result[key] = value
        
    print(result)
    
if __name__ == '__main__':

    reference_path = "E:\\WorkPlace\\Software\\data\\person_reid\\test_reference"

    query_image_path = "E:\\WorkPlace\\Software\\data\\person_reid\\test_query\\016246.jpg"
    query(query_image_path, reference_path)