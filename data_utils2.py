#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""
import re
import os
import pickle
import jieba
import shutil


def zero_digits(s):
    """将0~9数字字符统一用"0"字符取代
    """
    return re.sub('\d', '0', s)


def load_sentences(path, lower, zeros):
    """加载数据，将数据转为：
    [[[字, 标签], [字, 标签]...],   # 第一句
     [...],  # 第二句
     ...
     ]
    """
    sentences = []
    sentence = []

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            line = zero_digits(line) if zeros else line  # 数字转换为"0"
            line = line.lower() if lower else line  # 是否小写化

            if not line:  # 如果是空行（新句子）
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            else:  # 如果不是空行
                word = line.split(" ")
                assert len(word) == 2, print([word[0]])  # 确保切分后长度为2，[词，tag]
                sentence.append(word)

        if len(sentence) > 0:
            sentences.append(sentence)

    return sentences


def iob2(tags):
    """检查标签是否规范
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def update_tag_scheme(sentences):
    """检查标签，转为iobes标注
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]  # 标签序列
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('IOB标签有误，请检查 {}:\n{}'.format(i, s_str))

        new_tags = iob_iobes(tags)
        for word, new_tag in zip(s, new_tags):
            word[-1] = new_tag


def load_data(config):
    # 载入数据集
    train_sentences = load_sentences(config["train_file"], config["lower"], config["zeros"])
    dev_sentences = load_sentences(config["dev_file"], config["lower"], config["zeros"])
    test_sentences = load_sentences(config["test_file"], config["lower"], config["zeros"])

    # 修正语料标注格式（IOB→IOBES）
    update_tag_scheme(train_sentences)
    update_tag_scheme(dev_sentences)
    update_tag_scheme(test_sentences)

    return train_sentences, dev_sentences, test_sentences


def create_dic(sentences, lower=False):
    """创建词典（词——词频）
    """

    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]  # 字母转小写，抽取所有的word
    tags = [[x[-1] for x in s] for s in sentences]  # 标签列表

    num_words = sum(len(x) for x in words)
    num_tags = sum(len(x) for x in tags)
    print("词总数: {}".format(num_words))
    print("标签总数: {}".format(num_tags))
    assert num_words == num_tags, print("词与标签数量不等!")

    words_fre_dic = {}
    for sen_word in words:
        for word in sen_word:
            if word not in words_fre_dic:
                words_fre_dic[word] = 1
            else:
                words_fre_dic[word] += 1

    words_fre_dic["<PAD>"] = 10000001
    words_fre_dic['<UNK>'] = 10000000
    print("词种类数：{}".format(len(words_fre_dic)))

    tags_fre_dic = {}
    for sen_tag in tags:
        for tag in sen_tag:
            if tag not in tags_fre_dic:
                tags_fre_dic[tag] = 1
            else:
                tags_fre_dic[tag] += 1
    print("标签种类数：{}".format(len(tags_fre_dic)))

    return words_fre_dic, tags_fre_dic


def create_mapping(words_fre_dic):
    """创建词与编号的映射字典
    """

    sorted_items = sorted(words_fre_dic.items(), key=lambda x: (-x[1], x[0]))  # 降序排列
    id_to_word = {i: v[0] for i, v in enumerate(sorted_items)}  # {编号:词}
    word_to_id = {v: k for k, v in id_to_word.items()}  # {词: 编号}

    return word_to_id, id_to_word  # 返回{词: 编号}，{编号: 词}的映射字典


def augment_with_pretrained(dictionary, ext_emb_path):
    """将预训练embedding里的词添加到{词: 词频}字典里
    """
    print('加载预训练好的词向量...')
    assert os.path.isfile(ext_emb_path)

    words_pretrained = set()  # 预训练词
    for line in open(ext_emb_path):
        words_pretrained.add(line.rstrip().split()[0].strip())

    count = 0
    for word in words_pretrained:
        if word not in dictionary:
            count += 1
            dictionary[word] = 0  # 即训练集中该词的词频为0

    print("词表新增加 {} 种词,现有 {} 种词.".format(count, len(dictionary)))

    return dictionary


def create_maps(train_sentences, config):
    if not os.path.isfile(config["map_file"]):  # 创建新的maps
        words_dic_train, tags_dic_train = create_dic(train_sentences, config["lower"])  # 生成训练集{词: 词频}字典
        tag_to_id, id_to_tag = create_mapping(tags_dic_train)  # 创建标签与编号映射字典 {标签: 编号}, {编号: 标签}

        # 创建词与编号的映射字典 {词: 编号}, {编号: 词}
        if config["pre_emb"]:
            dic_add_pre = augment_with_pretrained(words_dic_train.copy(), config["emb_file"])  # 预训练词
            word_to_id, id_to_word = create_mapping(dic_add_pre)
        else:
            word_to_id, id_to_word = create_mapping(words_dic_train)

        with open(config["map_file"], "wb") as f:
            pickle.dump([word_to_id, id_to_word, tag_to_id, id_to_tag], f)  # 保存词和标签的编号映射

    else:  # 直接读取已有的maps
        with open(config["map_file"], "rb") as f:
            word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

    return word_to_id, id_to_word, tag_to_id, id_to_tag


def get_seg_features(string):
    """结巴分词，获取分词特征
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)  # 如果词长是1，seg_feature==0
        else:
            tmp = [2] * len(word)  # 如果词长>1，用1表示开头，用3表示结尾，用2表示中间
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature  # 所以seg_feature的长度仍然和字符串长度相同


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """得到N个句子的 [[词列表], [词编号], [分词特征编号], [tag编号]]
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def make_path(config):
    """生成路径
    """
    if not os.path.isdir(config["result_path"]):
        os.makedirs(config["result_path"])
    if not os.path.isdir(config["ckpt_path"]):
        os.makedirs(config["ckpt_path"])
    if not os.path.isdir(config["log_path"]):
        os.makedirs(config["log_path"])


def clean(config):
    """清空无关文件
    """
    if os.path.isfile(config["map_file"]):
        os.remove(config["map_file"])

    if os.path.isdir(config["ckpt_path"]):
        shutil.rmtree(config["ckpt_path"])

    if os.path.isdir(config["result_path"]):
        shutil.rmtree(config["result_path"])

    if os.path.isfile(config["config_file"]):
        os.remove(config["config_file"])

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")


if __name__ == "__main__":
    pass
