#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

num_chars = 10000
char_dim = 300
initializer = initializers.xavier_initializer()
num_segs = 4
seg_dim = 20

char_inputs = [[0, 1, 3]]
seg_inputs = [[0, 1, 2]]


def embedding_layer():
    """
    :param char_inputs: 句子的one-hot编码
    :param seg_inputs: segmentation feature
    :param config: wither use segmentation feature
    :return: [1, num_steps, embedding size],
    """

    embedding = []
    with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
        char_lookup = tf.get_variable(
            name="char_embedding",
            shape=[num_chars, char_dim],  # 词种类数，词向量维度
            initializer=initializer)  # 初始化

        embedding.append(tf.nn.embedding_lookup(char_lookup, char_inputs))

        with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
            seg_lookup = tf.get_variable(
                name="seg_embedding",
                shape=[num_segs, seg_dim],
                initializer=initializer)
            embedding.append(tf.nn.embedding_lookup(seg_lookup, seg_inputs))

        embed = tf.concat(embedding, axis=-1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = sess.run(embed)
        emb_weights = sess.run(char_lookup.read_value())
        print(x.shape)
        print(emb_weights.shape)

    return embed


embedding_layer()


if __name__ == "__main__":
    pass
