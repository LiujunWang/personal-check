import input_data
import tensorflow as tf
import numpy as np
import os


slim = tf.contrib.slim
# 产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev) 


# 设置L2正则的weight_decay, 标准差默认值0.1
def inception_v3_arg_scope(weight_decay = 0.00004, stddev = 0.1, 
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

def inception_v3_base(inputs, scope = None):


    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        #对三个参数设置默认值
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):

            net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
            #输出149*149*32数据
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            #输出147 x 147 x 32
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            #输出73 x 73 x 64
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            #输出73 x 73 x 80.
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            #输出71 x 71 x 192.
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
            #输出35 x 35 x 192
            # 设置所有模块组的默认参数,将所有卷积层、最大池化、平均池化层步长都设置为1
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出35 x 35 x 256
                with tf.variable_scope('Mixed_5c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出35 x 35 x 288
                with tf.variable_scope('Mixed_5d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出35 x 35 x 288
                #第二个inception模块
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        #图片会被压缩
                        branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2,padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                        # 图片被压缩
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2,padding = 'VALID', scope = 'Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',scope='MaxPool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], 3)
                    #输出17 x 17 x 768
                with tf.variable_scope('Mixed_6b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出17 x 17 x 768
                with tf.variable_scope('Mixed_6c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出17 x 17 x 768
                with tf.variable_scope('Mixed_6d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出17 x 17 x 768
                with tf.variable_scope('Mixed_6e'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出17 x 17 x 768
                # 第三个inception模块组包含了三个inception module
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        # 压缩图片
                        branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    # 池化层不会对输出通道数产生改变
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], 3)
                    #输出8 x 8 x 1280
                with tf.variable_scope('Mixed_7b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'): 
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出8 x 8 x 2048
                with tf.variable_scope('Mixed_7c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    #输出8 x 8 x 2048
                    return net


def inception_v3(inputs, dim_vector = 900, is_training = False, 
                 dropout_keep_prob = 1, # 节点保留比率
                 prediction_fn = slim.softmax, # 最后用来分类的函数
                 reuse = tf.AUTO_REUSE, # 是否对网络和Variable进行重复使用
                 scope = 'InceptionV3'):# 包含函数默认参数的环境
    # 定义参数默认值
    with tf.variable_scope(scope, 'InceptionV3', [inputs, dim_vector], reuse = reuse) as scope:
        # 定义标志默认值
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # 用定义好的函数构筑整个网络的卷积部分
            net = inception_v3_base(inputs, scope = scope)
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope='AvgPool_1a_8x8')

                net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
                logits = slim.conv2d(net, dim_vector, [1, 1], activation_fn = None, normalizer_fn = None, scope='Conv2d_1c_1x1')
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                return logits


def dense_to_one_hot(label_list, num_examples):
    
    length = len(label_list)
    result_label = np.zeros((length, num_examples))
    for x in range(length):
        temp = label_list[x]
        result_label[x][temp] = 1

    return np.array(result_label)

if __name__ == "__main__":

    learning_rate = 0.0001

    batch_size = 4

    capacity = 12

    dim_vector = 900

    epochs = 2
    '''
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    '''
    xml_file_path = "..\\doc\\train.xml"

    train_path = "..\\train"

    train_tfRecords_path = "train.tfrecords"

    test_tfRecords_path = "test.tfrecords"

    if not (os.path.isfile(train_tfRecords_path) and os.path.isfile(test_tfRecords_path)):
        print("tfrecords file开始生成...")
        input_data.create_record(xml_file_path, train_path)
    # 创建训练文件队列,不限读取的数量
    train_filename_queue = tf.train.string_input_producer(["train.tfrecords"], shuffle = True, num_epochs = epochs)#重复多少次并且在每个epoch内打乱顺序

    train_image, train_label = input_data.decode_file(train_filename_queue, batch_size = batch_size, capacity = capacity)

    train_image = tf.cast(train_image, tf.float32)
	
    '''
    placeholder_img = tf.placeholder("float", shape=[None, 299, 299, 3], name="placeholder_img")
    placeholder_lab = tf.placeholder("float", shape=[None, dim_vector], name="placeholder_lab")
    keep_prob = tf.placeholder("float", name="keep_prob")
	'''
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    count = 0
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "..\\Model\\model.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        with slim.arg_scope(inception_v3_arg_scope()) as scope:

            try:
                while not coord.should_stop():
                    train_image_value, train_label_index = sess.run([train_image, train_label])
                    train_label_value = dense_to_one_hot(train_label_index,dim_vector)

                    predictions = inception_v3(train_image_value, is_training = True, dim_vector = dim_vector, reuse = tf.AUTO_REUSE, dropout_keep_prob = 0.8)

                    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(train_label_value,1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = train_label_value, logits = predictions))
                    total_loss = slim.losses.get_total_loss()
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    train_op = slim.learning.create_train_op(total_loss, optimizer)
                    if count % 500 == 0:

                        loss_value = loss.eval()
                        accuracy_value = accuracy.eval()
                        print(loss_value)
                        print(accuracy_value)
                        
                    count = count + 1
                    
                    slim.learning.train(train_op, "..\\Model\\model.ckpt")
            except tf.errors.OutOfRangeError:
                print('Done reading Training comes to end')
                print('==========================================')
            finally:
                coord.request_stop()
                #print("Model saved in file:", saver_path)
                
            #coord.request_stop()
            coord.join(threads)



    

        

    
        


    
    
    
    
    
    