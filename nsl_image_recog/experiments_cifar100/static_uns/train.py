import numpy as np
import tensorflow as tf
from loss import loss2
from cifar100_input import *
from architecture import VGG
import os
import argparse
import shutil
import sys

def train(base_lr=1e-3, batch_sz=128, gpu_no=0):
    assert type(gpu_no) == int and gpu_no >= 0

    root_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = os.path.basename(root_path)

    log_path = os.path.join(root_path, '../../log_cifar10')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = os.path.join(log_path, folder_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    save_path = os.path.join(root_path, '../../model_cifar10')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, folder_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    n_class = 100
    batch_sz = batch_sz
    batch_test = 100
    max_epoch = 42500
    lr = base_lr
    momentum = 0.9
    is_training = tf.placeholder("bool")

    data_path = os.path.join(root_path, '../../../data/cifar-100')
    tr_images, tr_labels = distorted_inputs(data_path, batch_sz)
    te_images, te_labels = inputs(True, data_path, batch_test)
    images, labels = tf.cond(is_training, lambda: [tr_images, tr_labels], 
                                            lambda: [te_images, te_labels])
    # images = tf.constant(np.zeros((127,32,32,3)), dtype=np.float32)

    vgg = VGG()
    vgg.build(images, n_class, is_training)

    fit_loss = loss2(vgg.score, labels, n_class, 'c_entropy')
    loss_op = fit_loss
    reg_loss_list = tf.losses.get_regularization_losses()
    if len(reg_loss_list) != 0:
        reg_loss = tf.add_n(reg_loss_list)
        loss_op += reg_loss

    lr_ = tf.placeholder("float")

    # update_op = tf.train.AdamOptimizer(lr_).minimize(loss_op)
    update_op = tf.train.MomentumOptimizer(lr_, 0.9).minimize(loss_op)

    acc_op = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.to_int32(vgg.pred))))


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("")
    print("====================")
    print("Log will be saved to: " + log_path)
    print("")
    
    with open(os.path.join(log_path, 'log_test.txt'), 'w'):
        pass
    with open(os.path.join(log_path, 'log_train.txt'), 'w'):
        pass

    for i in xrange(max_epoch):

        if i == 20000:
            lr *= 0.1
        elif i == 30000:
            lr *= 0.1
        elif i == 37500:
            lr *= 0.1

        fit, reg, acc, _ = sess.run([fit_loss, reg_loss, acc_op, update_op], 
                                            {lr_: lr, is_training: True})

        if i % 100 == 0 and i != 0:
            print('====iteration_%d: fit=%.4f, reg=%.4f, acc=%.4f' 
                % (i, fit, reg, acc))
            with open(os.path.join(log_path, 'log_train.txt'), 'a') as train_acc_file:
                train_acc_file.write('====iteration_%d: fit=%.4f, reg=%.4f, acc=%.4f\n' 
                % (i, fit, reg, acc))


        if i % 500 == 0 and i != 0:
            n_test = 10000
            acc = 0.0
            for j in xrange(int(n_test/batch_test)):
                acc = acc + sess.run(acc_op, {is_training: False})
            acc = acc * batch_test / float(n_test)
            print('++++iteration_%d: test acc=%.4f' % (i, acc))
            with open(os.path.join(log_path, 'log_test.txt'), 'a') as test_acc_file:
                test_acc_file.write('++++iteration_%d: test acc=%.4f\n' % (i, acc))

        # if i%10000==0 and i!=0:
        #     tf.train.Saver().save(sess, os.path.join(save_path, str(i)))
    tf.train.Saver().save(sess, os.path.join(save_path, str(i)))
    
    n_test = 10000
    acc = 0.0
    for j in xrange(int(n_test/batch_test)):
        acc = acc + sess.run(acc_op, {is_training: False})
    acc = acc * batch_test / float(n_test)
    print('++++iteration_%d: test acc=%.4f' % (i, acc))
    with open(os.path.join(log_path, 'log_test.txt'), 'a') as test_acc_file:
        test_acc_file.write('++++iteration_%d: test acc=%.4f\n' % (i, acc))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_lr', type=float, default=1e-1,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
    args = parser.parse_args()

    train(args.base_lr, args.batch_size)
