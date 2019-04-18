import input_data
import tensorflow as tf
import numpy as np
import os


slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev) 

# 设置L2正则的weight_decay, 标准差默认值0.1
def network_arg_scope(weight_decay = 0.1, stddev = 0.1, 
                           batch_norm_var_collection = 'moving_vars'):
    
    # 定义batch normalization（标准化）的参数字典
    batch_norm_params = {  
        'decay': 0.9997,  #定义参数衰减系数
        'epsilon': 0.001, #防止除以0
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': 
        {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    

    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                weights_regularizer = slim.l2_regularizer(weight_decay)): # 对[slim.conv2d, slim.fully_connected]自动赋值
  # 使用slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置
        with slim.arg_scope([slim.conv2d], 
            weights_initializer = trunc_normal(stddev), 
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:

            return sc # 最后返回定义好的scope

def dense_to_one_hot(label_list, num_examples):
    
    length = len(label_list)
    result_label = np.zeros((length, num_examples))
    for x in range(length):
        temp = label_list[x]
        result_label[x][temp] = 1

    return np.array(result_label)

def network(inputs, dim_vector = 64, dropout_keep_prob = 0.5, is_training = True,reuse = tf.AUTO_REUSE, scope = 'InceptionV3', prediction_fn = slim.softmax,):

    with tf.variable_scope(scope, 'InceptionV3', [inputs, dim_vector], reuse = reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):
                branch_0 = slim.conv2d(inputs, 32, [3, 3], scope='branch_0')
                #62*62*32
                branch_1 = slim.max_pool2d(branch_0, [2, 2], stride = 2, scope='branch_1')
                #31*31*32
                branch_2 = slim.conv2d(branch_1, 64, [3, 3],  scope='branch_2')
                #29*29*64
                branch_3 = slim.max_pool2d(branch_2, [3, 3], stride = 2, padding = 'VALID', scope='branch_3')
                #14*14*64

                branch_4 = slim.conv2d(branch_3, 128, [3, 3], scope='branch_4')
                #12*12*128
                branch_5 = slim.max_pool2d(branch_4, [2, 2], stride = 2, scope='branch_5')
                #6*6*128

                branch_6 = slim.conv2d(branch_5, 512, [3, 3], scope='branch_6')
                #4*4*256
                branch_7 = slim.max_pool2d(branch_6, [4, 4], padding = 'VALID', scope='branch_7')
                #1*1*512

                branch_8 = slim.dropout(branch_7, dropout_keep_prob, scope='branch_8')

                branch_9 = slim.fully_connected(branch_8, dim_vector, scope = 'branch_9')
                predictions = tf.squeeze(branch_9, [1, 2], name='SpatialSqueeze')

    return predictions

if __name__ == "__main__":

    Q = 10

    batch_size = 8

    capacity = 16

    dim_vector = 64

    epochs = 5

    xml_file_path = "..\\doc\\train.xml"

    train_path = "..\\train"

    train_tfRecords_path = "train.tfrecords"

    test_tfRecords_path = "test.tfrecords"

    if not (os.path.isfile(train_tfRecords_path) and os.path.isfile(test_tfRecords_path)):
        print("tfrecords file开始生成...")
        input_data.create_record(xml_file_path, train_path)
    # 创建训练文件队列,不限读取的数量
    train_filename_queue = tf.train.string_input_producer(["train.tfrecords"], shuffle = True, num_epochs = epochs)#重复多少次并且在每个epoch内打乱顺序

    train_image_fir, train_image_sec, train_label = input_data.decode_file(train_filename_queue, batch_size = batch_size, capacity = capacity)

    train_image_fir = tf.cast(train_image_fir, tf.float32)
    train_image_sec = tf.cast(train_image_sec, tf.float32)

    ones = tf.ones([dim_vector, 1], dtype=tf.float32)
    placeholder_fir = tf.placeholder("float", shape=[None, 64, 64, 3], name="placeholder_fir")
    placeholder_sec = tf.placeholder("float", shape=[None, 64, 64, 3], name="placeholder_sec")
    placeholder_lab = tf.placeholder("float", shape=[None, ], name="placeholder_lab")
    keep_prob = tf.placeholder("float", name="keep_prob")

    with slim.arg_scope(network_arg_scope()) as scope:

        net_fir = network(placeholder_fir, dim_vector = dim_vector, dropout_keep_prob = keep_prob)
        net_sec = network(placeholder_sec, dim_vector = dim_vector, dropout_keep_prob = keep_prob)

        distance_square = tf.matmul(tf.square(net_fir - net_sec), ones)
        distance_sqrt = tf.sqrt(distance_square)

        loss_fun_pos = distance_square / Q
        loss_fun_neg = Q * tf.exp(-2.77 * distance_sqrt / Q)

        loss = tf.reduce_sum(tf.where(tf.greater(placeholder_lab, 0), loss_fun_neg, loss_fun_pos))
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    count = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "..\\Model\\model.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        try:
            while not coord.should_stop():
                train_image_fir_value, train_image_sec_value, train_label_value = sess.run([train_image_fir, train_image_sec, train_label])
                if count % 500 == 0:
                    '''
                    net_fir_value = net_fir.eval(feed_dict = {
                    placeholder_fir : train_image_fir_value,
                    placeholder_sec : train_image_sec_value,
                    keep_prob : 1.0
                    })
                    '''
                    loss_fun_pos_value = loss_fun_pos.eval(feed_dict = {
                    placeholder_fir : train_image_fir_value,
                    placeholder_sec : train_image_sec_value,
                    keep_prob : 1.0
                    })

                    loss_fun_neg_value = loss_fun_neg.eval(feed_dict = {
                        placeholder_fir : train_image_fir_value,
                        placeholder_sec : train_image_sec_value,
                        keep_prob : 1.0
                        })

                    loss_value = loss.eval(feed_dict = {
                        placeholder_fir : train_image_fir_value,
                        placeholder_sec : train_image_sec_value,
                        placeholder_lab : train_label_value,
                        keep_prob : 1.0
                        })

                    #print(train_label_value)
                    
                    print(loss_fun_pos_value)
                    print(loss_fun_neg_value)
                    print(train_label_value)
                    print(loss_value)
                    #print(net_fir_value.shape)
               
                    
                count = count + 1
                
                train_step.run(feed_dict = {
                                        placeholder_fir : train_image_fir_value,
                                        placeholder_sec : train_image_sec_value,
                                        placeholder_lab : train_label_value,
                                        keep_prob : 0.8,
                                  })
                
        except tf.errors.OutOfRangeError:
            print('Done reading Training comes to end')
            print('==========================================')
            print('Now Testing begin')
            

        finally:
            
            saver_path = saver.save(sess, "..\\Model\\model.ckpt") 
            print("Model saved in file:", saver_path)
            
            coord.request_stop()
        #coord.request_stop()
        coord.join(threads)




