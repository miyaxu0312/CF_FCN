import os
import math
import numpy as np
from PIL import Image
import scipy.io
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage import filters
from skimage import measure,color
import skimage.morphology as sm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module

IMAGE_SIZE = 64
NUM_OF_CLASSESS = 256
epoch = 30
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("img_dir", "save_img/", "path to images directory")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string("mode", "train", "train/test")
batchsize = 9


def get_data(mode, img_path, label_path, img_size, patch_size):
    my_img_path = img_path
    my_label_path = label_path
    imgfile_num = 0
    max_file = 1998  # 13张图的patch 14*81
    recordfile_num = 0
    recordfile_name = '%s%s%s' % (mode, recordfile_num, ".tfrecords")
    temp = mode + "_tfrecord"
    recordfile_path = os.path.join(temp, recordfile_name)
    writer = tf.python_io.TFRecordWriter(recordfile_path)

    offset = 64
    offset_num = 54
    mode = str(mode)
    """
    number = 11
    if mode == 'train':
        number = 11
    elif mode == 'test':
        number = 7
    """
    list_img = os.listdir(my_img_path)
    list_img.sort()

    list_label = os.listdir(my_label_path)
    list_label.sort()
    
    
    for each_img in list_img:
        for each_label in list_label:
            """
            id = each_img[0:2]
            if mode == 'test':
                number = 7 
            elif mode == 'train':
                number = 11 
            img_id = '%s%s' % (id, each_img[number:])
            id = each_label[0:2]
            if mode == 'train':
                label_id = '%s%s' % (id, each_label[10:])
            elif mode == 'test':
                label_id = '%s.jpg' % id
            """
            if each_img == each_label:  # 找到每个图像对应的label图
                
                myimg = Image.open(os.path.join(my_img_path, each_img))
                mylabel = Image.open(os.path.join(my_label_path, each_label))

                myimg = myimg.crop([24, 0, 3456+24, 2368])
                mylabel = mylabel.crop([24, 0,3456+24, 2368])
                         
                xleft = 0
                yleft = 0
                width = int(offset)
                height = int(offset)
                
                for x in range(int(37)):
                    for y in range(int(54)):
                        box = (xleft, yleft, xleft + width, yleft + height)
                        img_patch = myimg.crop(box)
                        #img_patch = img_patch.resize([IMAGE_SIZE, IMAGE_SIZE])   #resize 224*224
                        #img_patch = img_patch.convert("RGB")
                        img_patch_bytes = img_patch.tobytes()
                        
                        label_patch = mylabel.crop(box)
                        #label_patch = label_patch.convert("1")
                        #label_patch = label_patch.resize([IMAGE_SIZE, IMAGE_SIZE])    #resize 224*224
                        
                        label_patch_bytes = label_patch.tobytes()

                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_patch_bytes])),
                                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_patch_bytes]))}))
                        writer.write(example.SerializeToString())
                        xleft = xleft + offset
                        imgfile_num = imgfile_num + 1
                    yleft = yleft + offset
                    xleft = 0
             
        if (imgfile_num < max_file):
            pass
        else:
            imgfile_num = 0
            recordfile_num = recordfile_num + 1
            recordfile_name = '%s%s%s' % (mode, recordfile_num, ".tfrecords")
            
            temp = mode + "_tfrecord"
            recordfile_path = os.path.join(temp, recordfile_name)
            writer = tf.python_io.TFRecordWriter(recordfile_path)
    writer.close()


def read_data(filename, batch_size, flag):
    filename_queue = tf.train.string_input_producer(filename, shuffle=flag)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={"label": tf.FixedLenFeature([], tf.string),
                                                                     "img": tf.FixedLenFeature([], tf.string)})
    train_img = tf.decode_raw(features["img"], tf.uint8)
    train_img = tf.reshape(train_img, [IMAGE_SIZE, IMAGE_SIZE, 3])

    train_label = tf.decode_raw(features["label"], tf.uint8)
    train_label = tf.reshape(train_label, [IMAGE_SIZE, IMAGE_SIZE, 1])
    
    train_img = tf.cast(train_img, tf.float32)
    train_label = tf.cast(train_label, tf.int32)

    img_batch, label_batch = tf.train.batch([train_img, train_label], batch_size=batch_size)
    return img_batch, label_batch


def convLayer(x, kHeight, kWidth, featureNum, name):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    #w = get_variable([kHeight, kWidth, channel, featureNum], name=name + "_w")
    #b = bias_variable([featureNum], name=name + "_b")
    w = tf.get_variable(name + "_w", shape=[kHeight, kWidth, channel, featureNum])
    tf.summary.histogram(name + "_w", w)

    b = tf.get_variable(name + "_b", shape=[featureNum])
    tf.summary.histogram(name + "_b", b)

    featureMap = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    out = tf.nn.bias_add(featureMap, b)
    return tf.nn.relu(out, name=name + "_relu")


def atrous_convLayer(x, kHeight, kWidth, featureNum, rate, name):
    """atrous_convlutional"""
    channel = int(x.get_shape()[-1])
    #w = get_variable([kHeight, kWidth, channel, featureNum], name=name + "_w")
    #b = bias_variable([featureNum], name=name + "_b")
    w = tf.get_variable(name + "_w", shape=[kHeight, kWidth, channel, featureNum], initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name + "_w", w)
    initial = tf.constant(0.1, shape=[featureNum])
    b = tf.get_variable(name + "_b", initializer=initial)
    tf.summary.histogram(name + "_b", b)
    featureMap = tf.nn.atrous_conv2d(x, w, rate,padding="SAME")
    out = tf.nn.bias_add(featureMap, b)
    return tf.nn.relu(out, name=name + "_relu")


def my_net(image, keep_prob):
    with tf.variable_scope("mynet"):
        conv1_1 = convLayer(image, 3, 3, 64, "conv1_1")
        conv1_2 = convLayer(conv1_1, 3, 3, 64, "conv1_2")
        atrous_conv1 = atrous_convLayer(conv1_2, 3, 3, 64, 1, "atrous_conv1")

        conv2_1 = convLayer(atrous_conv1, 3, 3, 128, "conv2_1")
        conv2_2 = convLayer(conv2_1, 3, 3, 128, "conv2_2")
        atrous_conv2 = atrous_convLayer(conv2_2, 3, 3, 128, 2, "atrous_conv2")

        conv3_1 = convLayer(atrous_conv2, 3, 3, 256, "conv3_1")
        conv3_2 = convLayer(conv3_1, 3, 3, 256, "conv3_2")
        conv3_3 = convLayer(conv3_2, 3, 3, 256, "conv3_3")

        #conv3_4 = convLayer(conv3_3, 3, 3, 256, "conv3_4")
        atrous_conv3 = atrous_convLayer(conv3_3, 3, 3, 256, 3, "atrous_conv3")
        
        conv4_1 = convLayer(atrous_conv3, 3, 3, 512, "conv4_1")
        conv4_2 = convLayer(conv4_1, 3, 3, 512, "conv4_2")
        conv4_3 = convLayer(conv4_2, 3, 3, 512, "conv4_3")
        #conv4_4 = convLayer(conv4_3, 3, 3, 512, "conv4_4")
        atrous_conv4 = atrous_convLayer(conv4_3, 3, 3, 512, 5, "atrous_conv4")

        conv5_1 = convLayer(atrous_conv4, 3, 3, 512, "conv5_1")
        conv5_2 = convLayer(conv5_1, 3, 3, 512, "conv5_2")
        conv5_3 = convLayer(conv5_2, 3, 3, 512, "conv5_3")
        #conv5_4 = convLayer(conv5_3, 3, 3, 512, "conv5_4")
        atrous_conv5 = atrous_convLayer(conv5_3, 3, 3, 512, 7, "atrous_conv5")

        conv6 = convLayer(atrous_conv5, 7, 7, 1024, "conv6")
        relu_dropout6 = tf.nn.dropout(conv6, keep_prob=keep_prob)
        
        conv7 = convLayer(relu_dropout6, 1, 1, 1024, "conv7")
        relu_dropout7 = tf.nn.dropout(conv7, keep_prob=keep_prob)

        W8 = weight_variable([1, 1, 1024, NUM_OF_CLASSESS], name="W8")
        b8 = bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = conv2d_basic(relu_dropout7, W8, b8)
        #output = CrfRnnLayer(conv8, image, 9, image_dims=(IMAGE_SIZE, IMAGE_SIZE),num_classes=NUM_OF_CLASSESS, theta_alpha=180.,
            #theta_beta=3.,theta_gamma=3.,num_iterations=8,name='crfrnn')

        annotation_pred = tf.argmax(conv8, axis=3, name="myannotation")

    return tf.expand_dims(annotation_pred, dim=3), conv8


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        weight= tf.Variable(initial)
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.05)(weight))
    else:
        weight=tf.get_variable(name, initializer=initial)
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.05)(weight))
    return weight
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


def CrfRnnLayer(upscore, img_input, batchsize, image_dims, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations, name):
    spatial_ker_weights = tf.get_variable('spatial_weight', shape=[num_classes, num_classes], dtype=tf.float32, initializer =tf.contrib.layers.xavier_initializer())

    # Weights of the bilateral kernel
    bilateral_ker_weights = tf.get_variable('bilateral_weight', shape=[num_classes, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    # Compatibility matrix
    compatibility_matrix = tf.get_variable('compatible_weight', shape=[num_classes, num_classes], initializer=tf.contrib.layers.xavier_initializer())

    c, h, w = num_classes, image_dims[0], image_dims[1]
    all_ones = np.ones((c, h, w), dtype=np.float32)

    q_values_list = []
    for j in range(batchsize):
        my_upscore = tf.unstack(upscore, num=batchsize, axis=0)
        my_img_input = tf.unstack(img_input, num=batchsize, axis=0)

        q_values = tf.transpose(my_upscore[j], perm=(2, 0, 1))
        rgb = tf.transpose(my_img_input[j], perm=(2, 0, 1))
        unaries = q_values
        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False, theta_gamma=theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)

        for i in range(num_iterations):
            softmax_out = tf.nn.softmax(q_values, dim=0)
            
            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False, theta_gamma=theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals
            # Weighting filter outputs
            message_passing = (tf.matmul(spatial_ker_weights, tf.reshape(spatial_out, (c, -1))) +
                           tf.matmul(bilateral_ker_weights, tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise
            q_values2 = tf.transpose(tf.reshape(q_values, (c, h, w)), perm=(1, 2, 0))
        q_values_list.append(q_values2)

    my_q_values = tf.stack(q_values_list, axis=0)

    return tf.reshape(my_q_values,[batchsize, w, h, num_classes])


def evaluation_np(prediction, label, mywidth, myheight):
    mypredict = prediction
    mylabel = label
    width = mywidth
    height = myheight
    tp_num = 0
    tn_num = 0
    fp_num = 0
    fn_num = 0


    zeros_mylabel = np.zeros_like(mylabel)
    zeros_mypredict = np.zeros_like(mypredict)

    transfer_label = (mylabel == zeros_mylabel)    # ture 1 is the backgroud, false 0 is the vessel
    transfer_predict = (mypredict == zeros_mypredict)
    
    ones_like_mylabel = np.ones_like(transfer_label)
    zeros_like_mylabel= np.zeros_like(transfer_label)
    ones_like_mypredict = np.ones_like(transfer_predict)
    zeros_like_mypredict = np.zeros_like(transfer_predict)
   

    tp_op = np.logical_and(
        (transfer_label == zeros_like_mylabel), 
        (transfer_predict == zeros_like_mypredict)
      )
    tp_matrix = tp_op.getA()
    for i in range(width):
        for j in range(height):
            if str(tp_matrix[i][j]) == 'True':
                tp_num = tp_num + 1
   
    tn_op = np.logical_and(
        (transfer_label == ones_like_mylabel), 
        (transfer_predict == ones_like_mypredict)
      )

    tn_matrix = tn_op.getA()
   
    for i in range(width):
        for j in range(height):
            if str(tn_matrix[i][j]) == 'True':
                tn_num = tn_num + 1

    fp_op = np.logical_and(
        (transfer_label == ones_like_mylabel), 
        (transfer_predict == zeros_like_mypredict)
      )
    fp_matrix = fp_op.getA()
    for i in range(width):
        for j in range(height):
            if str(fp_matrix[i][j]) == 'True':
                fp_num = fp_num + 1

    fn_op = np.logical_and(
        (transfer_label == zeros_like_mylabel), 
        (transfer_predict == ones_like_mypredict)
      )
    fn_matrix = fn_op.getA()
    for i in range(width):
        for j in range(height):
            if str(fn_matrix[i][j]) == 'True':
                fn_num = fn_num + 1

    return tp_num, tn_num, fp_num, fn_num


def get_sp(tp, tn, fp, fn):
    sp = float(tn)/(float(tn) + float(fp))
    return sp


def get_se(tp, tn, fp, fn):
    se = float(tp)/(float(tp) + float(fn))
    return se


def get_accuracy(tp, tn, fp, fn):
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    return accuracy


def img_to_matrix(myimg):
    #wid, hei = myimg.size
    data = myimg.getdata()
    data = np.matrix(data, dtype='int')
    mymatrix = np.reshape(data,(54*64, 37*64))
    return mymatrix

def post_process(img, path, epoch_num, vessel_id):
    data = img.convert('1')
    data_close = sm.closing(data, sm.square(2))
    data_median = filters.median(np.array(data_close), disk(1))
    data_remove = sm.remove_small_objects(np.array(data_median), min_size=100, connectivity=2)
    post_process_img = Image.fromarray(data_remove.astype(np.uint8))
    post_process_img = post_process_img.convert('1')
    save_name = "posted_e%sn%s.png" % (epoch_num, vessel_id)
    save_path = os.path.join(path, save_name)
    post_process_img.save(save_path)
    return post_process_img

def test(test_tf_num,test_tf_list,test_img,epoch_num):
    #offset_num = int(imgsize / cropsize)
    #model_path = tf.train.latest_checkpoint('logs/')
    #ckpt = tf.train.get_checkpoint_state('')
    #if ckpt and ckpt.model_checkpoint_path:
    #saver.restore(sess, model_path)
    #print("Model restored...")
    for i in range(0, test_tf_num):
        tf_name = 'test' + str(i) + '.tfrecords'
        temp_path = os.path.join("test_tfrecord", tf_name)
        test_tf_list.append(temp_path)
    my_test_img, my_test_label = read_data(test_tf_list, batchsize,False)  # 获取tfrecord文件里的训练图和其label的batch

    test_iter = test_img * 54*37/ batchsize
    #subimg_max = int((imgsize / cropsize) * (imgsize / cropsize) / batchsize)  # the num of the paste time of the same image
    sub_img_num = 0
    col_num=0
    row_num=0
    my_temp_size = 64
    xleft = int(0)
    yleft = int(0)
    xright = int(my_temp_size)
    yright = int(my_temp_size)
    box = [xleft, yleft, xright, yright]
    myvesslimg = Image.new("L", [my_temp_size * 54, my_temp_size * 37])
    myvesslimg_masked = Image.new("L", [my_temp_size * 54, my_temp_size * 37])
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    name_num=1
    my_acc = np.zeros([test_img, 1])
    my_se = np.zeros([test_img, 1])
    my_sp = np.zeros([test_img, 1])
    my_acc2 = np.zeros([test_img, 1])
    my_se2 = np.zeros([test_img, 1])
    my_sp2 = np.zeros([test_img, 1])
    vessel_num = 9
    num_num=10
    txt_file = open('acc.txt', "a+")
    txt_file2 = open('acc_post.txt', "a+")
    f_tp = open('tp.txt', 'a+')
    f_tn = open('tn.txt', 'a+')
    f_fp = open('fp.txt', 'a+')
    f_fn = open('fn.txt', 'a+')
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    my_flag=False
    result_num = 0
    sep = ','
    for num in range(1, int(test_iter)+1):  # 1 epoch  30
        tf.train.start_queue_runners(sess=sess)
        [test_img_batch, test_label_batch] = sess.run([my_test_img, my_test_label])
        feed_dict_test = {image: test_img_batch, label: test_label_batch, keep_probability: 1.0}

        my_predict_test = sess.run(mypredict, feed_dict=feed_dict_test)
        my_loss = sess.run(loss, feed_dict=feed_dict_test)
        print("---TEST---epoch:%s  iter:%s / %s    loss:%.5s" % (epoch_num, num, test_iter, my_loss))
        
        row_num = row_num + 1
        my_test_predict = np.squeeze(my_predict_test, axis=3)
        for batch_num in range(batchsize):                       # 1 batchsize
            patch_img = Image.fromarray(my_test_predict[batch_num].astype(np.uint8))
            myvesslimg.paste(patch_img, box)
            xleft = int(xleft + my_temp_size)
            xright = int(xright + my_temp_size)
            box = [xleft, yleft, xright, yright]
        if (row_num % 6 == 0):
            xleft = int(0)
            yleft = int(yleft + my_temp_size)
            xright = int(my_temp_size)
            yright = int(yright + my_temp_size)
            row_num=0
            col_num=col_num+1
        #print(col_num)
        box = [xleft, yleft, xright, yright]
        if (col_num % 37 == 0 and col_num!= 0):  # each 9 batchsize, another new image
            col_num = 0
            row_num=0
            #vessel_num = vessel_num + 1
            if my_flag:
                vessel_id = str(vessel_num)                    
            else:
                vessel_id = "%s%s" % ('0', str(vessel_num))
            if name_num == 1:
                vessel_id = "%s%s" % (vessel_id, '_dr')
                name_num=2
            elif name_num == 2:
                vessel_id = "%s%s" % (vessel_id, '_g')
                name_num=3
            elif name_num == 3:
                vessel_id = "%s%s" % (vessel_id, '_h')
                vessel_num = vessel_num + 1
                name_num=1   
                my_flag=True
            list_mask = os.listdir(test_mask_path)
            list_mask.sort()
            for each_mask in list_mask:
                mask_id = each_mask[:-9]
                
                if vessel_id == str(mask_id):
                    mymask = Image.open(os.path.join(test_mask_path, each_mask))
                    mymask = mymask.convert("1")
                    mymask = mymask.crop([24, 0, 3456+24, 2368])
                    #myvesslimg = myvesslimg.resize([3456,2368])
                    mymask_arr = np.array(mymask)
                    myvesslimg_masked_array = np.multiply(np.array(myvesslimg), mymask_arr)
                    myvesslimg_masked = Image.fromarray(myvesslimg_masked_array)
                    myvesslimg_masked.save("%s%se%sn%s%s" % ('masked_imgs/', 'masked_vessel', epoch_num, vessel_id, '.png'))
            list_label = os.listdir(test_label_path)
            list_label.sort()
            for each_label in list_label:
                label_id = each_label[:-4]
                if vessel_id == str(label_id):
                    mylabel = Image.open(os.path.join(test_label_path, each_label))
                    mylabel = mylabel.convert("1")
                    mylabel = mylabel.crop([24, 0, 3456 + 24, 2368])
                    mylabel_matrix = img_to_matrix(mylabel)
                    
                    post_processed_img = post_process(myvesslimg_masked, 'postprocessed_imgs/', epoch_num, vessel_id)

                    myvesslimg_masked = myvesslimg_masked.convert("1")
                    post_processed_img_matrix  = img_to_matrix(post_processed_img)
                    myvesslimg_masked_matrix = img_to_matrix(myvesslimg_masked)
                    tp, tn, fp, fn = evaluation_np(myvesslimg_masked_matrix, mylabel_matrix, 54*64, 37*64)
                    tp_list.append(str(tp))
                    tn_list.append(str(tn))
                    fp_list.append(str(fp))
                    fn_list.append(str(fn))
                    my_se[result_num] = get_se(tp, tn, fp, fn)          
                    my_sp[result_num] = get_sp(tp, tn, fp, fn)       
                    my_acc[result_num] = get_accuracy(tp, tn, fp, fn)

                    tp2, tn2, fp2, fn2 = evaluation_np(post_processed_img_matrix, mylabel_matrix, 54*64, 37*64)
                    
                    my_se2[result_num] = get_se(tp2, tn2, fp2, fn2)          
                    my_sp2[result_num] = get_sp(tp2, tn2, fp2, fn2)    
                    my_acc2[result_num] = get_accuracy(tp2, tn2, fp2, fn2) 
                    
                    print("---TEST---No:%s/%s accuracy:%s" % (result_num, test_img, my_acc[result_num]))
                    print("---postprocessed---No:%s/%s accuracy:%s" % (result_num, test_img, my_acc2[result_num]))
                    print("tp:%.5s tn:%.5s  fp:%.5s fn:%.5s  se:%.5s  sp:%.5s" % (tp, tn, fp, fn, my_se[result_num], my_sp[result_num]))
            myvesslimg = Image.new("L", [my_temp_size * 54, my_temp_size * 37])
            myvesslimg_masked = Image.new("L", [my_temp_size * 54, my_temp_size * 37])
            xleft = int(0)
            yleft = int(0)
            xright = int(my_temp_size)
            yright = int(my_temp_size)
            box = [xleft, yleft, xright, yright]
        #tf.summary.scalar("test_accuracy", final_acc)
        summary_str = sess.run(summary_op, feed_dict=feed_dict_test)
        summary_writer_test.add_summary(summary_str, epoch_num)
    average_acc = np.mean(my_acc)
    average_se = np.mean(my_se)
    average_sp = np.mean(my_sp)
    average_acc2 = np.mean(my_acc2)
    average_se2 = np.mean(my_se2)
    average_sp2 = np.mean(my_sp2)
    print("Epoch:%s Average acc:%s\nse:%.8s  sp:%.8s" % (epoch_num, average_acc, average_se, average_sp))
    print("Epoch:%s postprocessed Average acc:%s\nse:%.8s  sp:%.8s" % (epoch_num, average_acc2, average_se2, average_sp2))
    mystring = ("Epoch:%s Average acc:%s\nse:%.8s  sp:%.8s" % (epoch_num, average_acc, average_se, average_sp))
    mystring2 = ("Epoch:%s Average acc:%s\nse:%.8s  sp:%.8s" % (epoch_num, average_acc2, average_se2, average_sp2))
    txt_file.write(mystring)
    txt_file2.write(mystring2)
    result_num = result_num + 1
    txt_file.close()
    txt_file2.close()
    
    epoch_num = epoch_num + 1
    summary_writer_train.close()
    summary_writer_test.close()
   


if __name__ == '__main__':
    '''
    用VGG神经网络解决眼底血管分割问题
    '''
    img_path = "my_dataset/hrf/train/img"
    label_path = "my_dataset/hrf/train/label"
    
    post_process_path = "postprocessed_imgs"
    
    test_img_path = "my_dataset/hrf/test/img"
    test_label_path = "my_dataset/hrf/test/label"
    test_mask_path = "my_dataset/hrf/test/mask"

    imgsize = int(576)
    patchsize = IMAGE_SIZE
    cropsize = 64
    batchsize = 9
    train_img = 144
    test_img = 21
    img_num = 81
    train_tf_list = []
    validation_tf_list = []
    test_tf_list = []

    train_tf_num = int(math.ceil(train_img * 54*37 / 1998))

    #validation_tf_num = int(math.ceil(validation_img * 81 / 1053))
    test_tf_num = int(math.ceil(test_img * 54*37 / 1998))

    sess = tf.InteractiveSession()
    get_data("train", img_path, label_path, 64, cropsize)                        # 生成tfrecord文件
    #get_data("validation", valid_img_path, valid_label_path, imgsize, cropsize)
    get_data("test", test_img_path, test_label_path, 64, cropsize)
    
    # read training data
    for i in range(0, train_tf_num):
        tf_name = 'train' + str(i) + '.tfrecords'
        temp_path = os.path.join("train_tfrecord", tf_name)
        train_tf_list.append(temp_path)
    #temp_path = os.path.join("train_tfrecord", 'train0.tfrecords')   
    #train_tf_list.append(temp_path)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    label = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="label")

    my_img, my_label = read_data(train_tf_list, batchsize, True)  # 获取tfrecord文件里的训练图和其label的batch
    #mypredict, logit = inference(image, keep_probability)
    mypredict, logit = my_net(image, keep_probability)

    tf.summary.image("input_image", image, max_outputs=batchsize)
    tf.summary.image("ground_truth", tf.cast(label, tf.uint8), max_outputs=batchsize)
    tf.summary.image("pred_annotation", tf.cast(mypredict, tf.uint8), max_outputs=batchsize)

    #temp_label = tf.cast(label, tf.int32)
    # temp_label = tf.reshape(temp_label, [batchsize, patchsize, patchsize, 1])

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.squeeze(label,squeeze_dims=[3]), name="entropy")))
    #loss = tf.reduce_mean(tf.square(label-mypredict))
    tf.summary.scalar("entropy", loss)
    
    #train_var = tf.trainable_variables()
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    #grads = optimizer.compute_gradients(loss, train_var)
    #my_train = optimizer.apply_gradients(grads)
    #tf.summary.scalar("grads", grads)
    my_train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    #mypredict2 = tf.cast(mypredict, tf.uint8)  # true为0像素,即背景像素

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    ckpt = tf.train.get_checkpoint_state('logs/')
    summary_writer_train = tf.summary.FileWriter(FLAGS.logs_dir+"/train", sess.graph)
    summary_writer_test = tf.summary.FileWriter(FLAGS.logs_dir+"/test", sess.graph)
    # 进行2000次迭代
    
    with tf.device("/gpu:0"):
        #model_path = tf.train.latest_checkpoint('logs/')
        #ckpt = tf.train.get_checkpoint_state('')
        #if ckpt and ckpt.model_checkpoint_path:
        #saver.restore(sess, model_path)
        #print("Model restored...")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
    i = 0
    iter_num = int(train_img * 54*37/ batchsize)              #30*81/9=180
    max_iteration = epoch * iter_num
    epoch_num = 0
    #print("----------test----------")
    #test(test_tf_num,test_tf_list,test_img,epoch_num)
    print("----------training----------")
    for my_num in range(max_iteration):
        with tf.device("/gpu:0"): 
            img_batch, label_batch = sess.run([my_img, my_label])
            feed_dict_train = {image: img_batch, label: label_batch, keep_probability: 0.7}

            _, my_loss = sess.run([my_train, loss], feed_dict=feed_dict_train)
           
            if (my_num % 10 == 0):
                summary_str = sess.run(summary_op, feed_dict=feed_dict_train) 
                summary_writer_train.add_summary(summary_str, my_num)
            if (my_num % iter_num == 0 and my_num != 0):
                each_iter = iter_num
            else:
                each_iter = my_num % iter_num
            #my_loss = sess.run(loss, feed_dict=feed_dict_train)
            #acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print("---TRAINING---epoch:%s  iter:%s / %s    loss:%s " % (epoch_num, each_iter, iter_num, my_loss))
        if (my_num % iter_num == 0 and my_num != 0):
        #if (my_num == 10):
            model_path = FLAGS.logs_dir + "model.ckpt" + str(epoch_num)
            print("----------storing----------")
            saver.save(sess, model_path)                                        #每迭代一个epoch保存一次当前模型
                                              
            print("----------test----------")
            test(test_tf_num,test_tf_list,test_img,epoch_num)