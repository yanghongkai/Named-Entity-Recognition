# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        # print("lr:{}".format(self.lr))
        # print("char_dim:{}".format(self.char_dim))
        # print("lstm_dim:{}".format(self.lstm_dim))
        # print("seg_dim:{}".format(self.seg_dim))
        # print("num_tags:{}".format(self.num_tags))
        # print("num_chars:{}".format(self.num_chars))
        # print("num_segs:{}".format(self.num_segs))
        # lr: 0.001
        # char_dim: 100
        # lstm_dim: 100
        # seg_dim: 0
        # num_tags: 34
        # num_chars: 1103
        # num_segs: 4

        self.global_step = tf.Variable(0, trainable=False)
        # print("global_step:{}".format(self.global_step))
        # "global_step:<tf.Variable 'Variable:0' shape=() dtype=int32_ref>"
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        # print("char_inputs:{}".format(self.char_inputs))
        # print("seg_inputs:{}".format(self.seg_inputs))
        # print("targets:{}".format(self.seg_inputs))
        # print("dropouts:{}".format(self.dropout))
        # char_inputs: Tensor("ChatInputs:0", shape=(?, ?), dtype = int32)
        # seg_inputs: Tensor("SegInputs:0", shape=(?, ?), dtype = int32)
        # targets: Tensor("SegInputs:0", shape=(?, ?), dtype = int32)
        # dropouts: Tensor("Dropout:0", dtype=float32)


        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        # print("lengths:{}".format(self.lengths))
        # lengths: Tensor("Sum:0", shape=(?, ), dtype = int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        # print("batch_size:{}".format(self.batch_size))
        # print("num_steps:{}".format(self.num_steps))
        # batch_size: Tensor("strided_slice:0", shape=(), dtype=int32)
        # num_steps: Tensor("strided_slice_1:0", shape=(), dtype=int32)

        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        # print("embedding:{}".format(embedding))
        # embedding: Tensor("char_embedding/concat:0", shape=(?, ?, 100), dtype = float32, device = / device: CPU:0)
        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)
        # print("lstm_inputs:{}".format(lstm_inputs))
        # lstm_inputs: Tensor("dropout/mul:0", shape=(?, ?, 100), dtype = float32)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # print("lstm_inputs:{}".format(lstm_outputs))
        # print("lstm_outputs:{}".format(lstm_outputs))
        # print("logits:{}".format(self.logits))
        # lstm_inputs: Tensor("concat:0", shape=(?, ?, 200), dtype = float32)
        # lstm_outputs: Tensor("concat:0", shape=(?, ?, 200), dtype = float32)
        # logits: Tensor("project/Reshape:0", shape=(?, ?, 34), dtype = float32)


        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)
        # print("loss:{}".format(self.loss))
        # loss: Tensor("crf_loss/Mean:0", shape=(), dtype=float32)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        # print("project_logits:{}".format(project_logits))
        # project_logits: Tensor("project/Reshape:0", shape=(?, ?, 34), dtype = float32)
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            # print("start_logits:{}".format(start_logits))
            # start_logits: Tensor("crf_loss/concat:0", shape=(?, 1, 35), dtype = float32)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            # print("pad_logits:{}".format(pad_logits))
            # pad_logits: Tensor("crf_loss/mul_1:0", shape=(?, ?, 1), dtype = float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            # print("logits:{}".format(logits))
            # logits: Tensor("crf_loss/concat_1:0", shape=(?, ?, 35), dtype = float32)
            logits = tf.concat([start_logits, logits], axis=1)
            # print("logits:{}".format(logits))
            # logits: Tensor("crf_loss/concat_2:0", shape=(?, ?, 35), dtype = float32)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            # print("log_likelihood:{}".format(log_likelihood))
            # log_likelihood: Tensor("crf_loss/sub_2:0", shape=(?, ), dtype = float32)
            # print("trans:{}".format(self.trans))
            # "trans:<tf.Variable 'crf_loss/transitions:0' shape=(35, 35) dtype=float32_ref>"
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        # print("is_train:{}".format(is_train))
        # is_train: True
        feed_dict = self.create_feed_dict(is_train, batch)
        # print("feed_dict:{}".format(feed_dict))
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        # print("evaluate_line:{}".format(inputs))
        # "[['请调出汇坤园北门2016年4月9日中午11:40的录像,四倍速回放'], " \
        # "[[25, 27, 28, 134, 731, 41, 29, 5, 4, 7, 2, 16, 17, 14, 8, 24, 9, 42, 33, 2, 2, 3, 14, 7, 6, 10, 11, 1, 108, 18, 19, 34, 36]], [[0, 1, 3, 1, 2, 3, 1, 3, 1, 2, 2, 3, 0, 0, 0, 0, 0, 1, 3, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3]], " \
        # "[[]]]"
        trans = self.trans.eval()
        # print("trans:{}".format(trans.shape))
        lengths, scores = self.run_step(sess, False, inputs)
        # print("lengths:{}".format(lengths.shape))
        # print("scores:{}".format(scores.shape))
        # trans: (35, 35)
        # lengths: (1,)
        # scores: (1, 33, 34)
        batch_paths = self.decode(scores, lengths, trans)
        # print("batch_paths:{}".format(batch_paths))
        'batch_paths:[[24, 32, 24, 32, 21, 7, 15, 9, 17, 25, 23, 26, 25, 6, 17, 25, 23, 26, 25, 23, 26, 8, 32, 24, 32, 24, 32, 24, 21, 19, 4, 33, 18]]'
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        # print("tags:{}".format(tags))
        "tags:['B-SPEED', 'S-ELOC', 'B-SPEED', 'S-ELOC', 'B-ETIME', 'B-SLOC', 'B-STIME', 'B-DAY', 'I-SPEED', 'E-SPEED', 'I-DAY', 'I-ROAD', 'E-SPEED', 'E-TYPE', 'I-SPEED', 'E-SPEED', 'I-DAY', 'I-ROAD', 'E-SPEED', 'I-DAY', 'I-ROAD', 'E-SLOC', 'S-ELOC', 'B-SPEED', 'S-ELOC', 'B-SPEED', 'S-ELOC', 'B-SPEED', 'B-ETIME', 'E-YEAR', 'I-ETIME', 'S-SLOC', 'B-YEAR']"
        return result_to_json(inputs[0][0], tags)
