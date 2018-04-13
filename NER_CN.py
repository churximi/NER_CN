#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：通用命名实体识别，基于字向量
时间：2018年04月07日14:20:44
"""
import json
import pickle
import tensorflow as tf
import numpy as np
from model import Model
from collections import OrderedDict
from utils import get_logger, save_model, test_ner, load_config
from data_utils2 import load_data, create_maps, prepare_dataset, make_path, clean
from data_utils import load_word2vec, input_from_line, BatchManager


def config_model():

    config = OrderedDict()
    config["train_file"] = "data/corpus/train.txt"
    config["dev_file"] = "data/corpus/dev.txt"
    config["test_file"] = "data/corpus/test.txt"

    config["ckpt_path"] = "ckpt"  # 模型存储路径
    config["log_path"] = "log"
    config["result_path"] = "result"
    config["log_file"] = "log/train.log"
    config["map_file"] = "data/maps.pkl"  # 映射字典
    config["config_file"] = "data/config.json"  # 配置文件

    config["max_epoch"] = 20  # 轮数
    config["batch_size"] = 32

    config["char_dim"] = 50  # 训练词向量维度
    config["seg_dim"] = 20  # Embedding size for segmentation, 0 if not used
    config["lstm_dim"] = 100  # LSTM隐藏单元数

    config["zeros"] = True  # 是否将数字替换为0
    config["lower"] = False
    config["pre_emb"] = True  # 是否使用预训练embedding
    config["emb_file"] = "data/word_embeddings.txt"

    config["clip"] = 5  # Gradient clip，用于处理梯度爆炸（<5.1)
    config["dropout_keep"] = 0.5  # (0, 1]，等于1时表示不采用dropout
    config["lr"] = 0.001  # 学习率（>0）
    config["optimizer"] = "adam"  # 可选adam, sgd, adagrad
    config["steps_check"] = 100  # steps per checkpoint

    return config


def evaluate(sess, model, name, data, id_to_tag, logger, config):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, config["result_path"])
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def do_train(config):
    train, dev, test = load_data(config)  # 加载数据
    word_to_id, id_to_word, tag_to_id, id_to_tag = create_maps(train, config)  # 创建或读取maps

    # 配置信息及保存
    config["num_chars"] = len(word_to_id)  # 词总数
    config["num_tags"] = len(tag_to_id)  # 标签总数
    with open(config["config_file"], "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 数据处理
    train_data = prepare_dataset(train, word_to_id, tag_to_id, config["lower"])
    dev_data = prepare_dataset(dev, word_to_id, tag_to_id, config["lower"])
    test_data = prepare_dataset(test, word_to_id, tag_to_id, config["lower"])

    print("train/dev/test 句子数：{} / {} / {}".format(len(train_data), len(dev_data), len(test_data)))

    # 分batch
    train_manager = BatchManager(train_data, config["batch_size"])
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    steps_per_epoch = train_manager.len_data  # 每个轮次的steps

    # 创建相关路径
    make_path(config)

    # logger
    logger = get_logger(config["log_file"])

    # GPU限制
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # 创建模型, 可以提供使用现有参数配置
        model = Model(config)

        ckpt = tf.train.get_checkpoint_state(config["ckpt_path"])  # 从模型路径获取ckpt
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):  # 现有模型
            logger.info("读取现有模型...")
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("新建模型...")
            sess.run(tf.global_variables_initializer())  # 不使用预训练的embeddings

            # 如果使用预训练的embeddings
            if config["pre_emb"]:
                emb_weights = sess.run(model.char_lookup.read_value())
                emb_weights = load_word2vec(config["emb_file"], id_to_word, config["char_dim"], emb_weights)
                sess.run(model.char_lookup.assign(emb_weights))
                logger.info("Load pre-trained embedding.")

        logger.info("开始训练...")
        loss = []
        for i in range(config["max_epoch"]):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)

                if step % config["steps_check"] == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))

                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger, config)
            if best:
                save_model(sess, model, config["ckpt_path"], logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger, config)


def evaluate_single(config):
    config = load_config(config["config_file"])
    logger = get_logger(config["log_file"])

    with open(config["map_file"], "rb") as f:
        word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = Model(config)

        logger.info("读取现有模型...")
        ckpt = tf.train.get_checkpoint_state(config["ckpt_path"])  # 从模型路径获取ckpt
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            line = input("请输入测试句子:\n")
            result = model.evaluate_line(sess, input_from_line(line, word_to_id), id_to_tag)
            print(result)
            entities = list(set([item["word"] for item in result["entities"]]))
            print(entities)


def evaluate_file(config):
    config = load_config(config["config_file"])
    logger = get_logger(config["log_file"])

    with open(config["map_file"], "rb") as f:
        word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

    fout = open("data/result_entities.json", "w+")

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = Model(config)

        logger.info("读取现有模型...")
        ckpt = tf.train.get_checkpoint_state(config["ckpt_path"])  # 从模型路径获取ckpt
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        with open("data/result.json") as f:
            for index, line in enumerate(f):
                if index % 1000 == 0:
                    print(index)

                data = json.loads(line.strip())
                answer = data["answers"][0]
                if not answer:
                    answer += " "
                result = model.evaluate_line(sess, input_from_line(answer, word_to_id), id_to_tag)
                entities = list(set([item["word"] for item in result["entities"]]))
                data["entity_answers"] = [entities]
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    fout.close()


def main():
    config = config_model()
    is_clean = False
    is_train = False
    if is_clean:
        clean(config)
    if is_train:
        do_train(config)

    # evaluate_single(config)
    # evaluate_file(config)


if __name__ == "__main__":
    main()
