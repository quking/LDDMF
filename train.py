import argparse

import numpy as np
import tensorflow as tf

from DataProcess import DataSetProcess


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-negNum', action='store', dest='negNum', default=4, type=int)
    parser.add_argument('-reg', action='store', dest='reg', default=1e-2)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=30, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=1024, type=int)
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()


class Model:
    def __init__(self, args):
        self.dataSet = DataSetProcess()
        self.cnt_lncRNA = self.dataSet.cnt_lncRNA
        self.cnt_disease = self.dataSet.cnt_disease
        self.embedding_size = 16
        self.deep_layers = [32, 16, 8]
        self.cross_layer_num = 4
        self.reg = args.reg
        self.negNum = args.negNum
        self.add_placeholders()
        self.add_embedding_matrix()
        self.add_model()
        self.add_loss()
        self.lr = args.lr
        self.add_train_step()
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize
        self.topK = args.topK

    def add_placeholders(self):
        self.lncRNA_index = tf.placeholder('int32', [None])
        self.disease_index = tf.placeholder('int32', [None])
        self.lable = tf.placeholder('float32', [None, 1])
        self.weights = {}
        self.weights['lncRNA'] = tf.get_variable(
            name='lncRNA_embedding',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[self.cnt_lncRNA, self.embedding_size]
        )
        self.weights['disease'] = tf.get_variable(
            name='disease_embedding',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[self.cnt_disease, self.embedding_size]
        )

    def add_embedding_matrix(self):
        self.lncRNA_embeddings = tf.nn.embedding_lookup(self.weights['lncRNA'], self.lncRNA_index)
        self.disease_embeddings = tf.nn.embedding_lookup(self.weights['disease'], self.disease_index)

    def add_model(self):
#----------------------deep Layer-----------------------------
        self.weights['layer_0'] = tf.get_variable(
            name='layer_0',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[2 * self.embedding_size, self.deep_layers[0]]
        )
        self.weights['bias_0'] = tf.get_variable(
            name='bias_0',
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            shape=[1, self.deep_layers[0]]
        )
        for i in range(1, len(self.deep_layers)):
            self.weights["layer_%d" % i] = tf.get_variable(
                name='layer_%d' % i,
                dtype=tf.float32,
                initializer=tf.glorot_normal_initializer(),
                shape=[self.deep_layers[i - 1], self.deep_layers[i]]
            )
            self.weights["bias_%d" % i] = tf.get_variable(
                name='bias_%d' % i,
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0),
                shape=[1, self.deep_layers[i]]
            )
        y_deep = tf.concat([self.lncRNA_embeddings, self.disease_embeddings], axis=1)
        for i in range(len(self.deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
            y_deep = tf.nn.relu(y_deep)
#----------------------------cross Feature
        y_cross = tf.concat([self.lncRNA_embeddings, self.disease_embeddings], axis=1)
        cross_layer_num = 4
        for i in range(cross_layer_num):
            self.weights["cross_layer_%d" % i] = tf.get_variable(
                name='cross_layer_%d' % i,
                dtype=np.float32,
                initializer=tf.glorot_normal_initializer(),
                shape=[2 * self.embedding_size, 1]
            )
            self.weights["cross_bias_%d" % i] = tf.get_variable(
                name="cross_bias_%d" % i,
                dtype=np.float32,
                initializer=tf.glorot_normal_initializer(),
                shape=[2 * self.embedding_size, 1]
            )
        y_cross = tf.reshape(y_cross, (-1, 2 * self.embedding_size, 1))
        y_cross_1 = y_cross
        for l in range(cross_layer_num):
            y_cross_1 = tf.tensordot(tf.matmul(y_cross, y_cross_1, transpose_b=True),
                                     self.weights["cross_layer_%d" % l], 1) + self.weights["cross_bias_%d" % l] + y_cross_1
        cross_network_out = tf.reshape(y_cross_1, (-1, 2 * self.embedding_size))
        distance = tf.keras.layers.Subtract()([self.lncRNA_embeddings, self.disease_embeddings])
        angle = tf.multiply(self.lncRNA_embeddings, self.disease_embeddings)
        predict_vector = tf.concat([cross_network_out, y_deep, distance, angle], axis=1)
        self.weights['layer_prediction'] = tf.get_variable(
            name='layer_prediction',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[self.deep_layers[-1] + self.embedding_size * 4, 1]
        )
        self.weights['bias_prediction'] = tf.get_variable(
            name='bias_prediction',
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            shape=[1, 1]
        )
        self.prediction = tf.add(tf.matmul(predict_vector, self.weights['layer_prediction']), self.weights['bias_prediction'])
        self.final = tf.sigmoid(self.prediction)

    def add_loss(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.lable)
        loss = tf.reduce_mean(loss)

        regular_decay = 0.01
        loss = loss + regular_decay * tf.reduce_mean(tf.reduce_sum(tf.square(self.lncRNA_embeddings), axis=1))
        loss = loss + regular_decay * tf.reduce_mean(tf.reduce_sum(tf.square(self.disease_embeddings), axis=1))
        loss = loss + tf.contrib.layers.l2_regularizer(regular_decay)(self.weights['layer_prediction'])
        for i in range(len(self.deep_layers)):
            loss = loss + tf.contrib.layers.l2_regularizer(regular_decay)(self.weights["layer_%d" % i])
        for i in range(self.cross_layer_num):
            loss += tf.contrib.layers.l2_regularizer(
                regular_decay)(self.weights["cross_layer_%d" % i])
        self.loss = loss

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        learning_rate = 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        tran_data = self.dataSet.use_data

        self.sess.run(tf.global_variables_initializer())
        train_data = self.dataSet.produce_negative_data(tran_data, 4, self.dataSet.cnt_lncRNA)
        for each_epoch in range(self.maxEpochs):
            temp = []
            train_data = train_data.sample(frac=1)
            begin = 0
            while (begin < len(train_data)):
                batch_data = train_data.iloc[begin:begin + self.batchSize]
                _, t1 = self.sess.run([self.train_op, self.loss],
                                 feed_dict={
                                     self.lncRNA_index: batch_data['ncRNA Symbol'],
                                     self.disease_index: batch_data['Disease Name'],
                                     self.lable: np.reshape(np.array(batch_data['lable']), newshape=[-1, 1])
                                 })
                begin = begin + self.batchSize
                temp.append(t1)
            print("epoch:%d train_loss:%lf" % (each_epoch, sum(temp) / len(temp)))
        print("save model to /checkPoint/model.ckpt")
        self.saver.save(self.sess, "./checkPoint/model.ckpt")


if __name__ == '__main__':
    main()
