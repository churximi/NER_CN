#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：通过计算候选实体词与问题的相似度，筛选出与问题相关性高的实体
时间：2018年04月07日00:21:41
"""

import json
import gensim
import numpy as np


def get_id_question():
    with open("data/测试集问题.json") as f:
        questions = json.load(f)
    id_question = {q["question_id"]: q["questions"][0] for q in questions}
    return id_question


def get_vocab():
    with open("data/vocab.txt") as f:
        vocab = [line.strip() for line in f]
    return vocab


def filter_entities(model, vocab, question, candidate_entities):
    answers, final_answers = [], []
    for entity in candidate_entities:
        similarity = 0
        if entity in vocab:
            if entity not in question:
                for word in question:
                    if word in vocab:
                        similarity += model.similarity(entity, word)
        else:
            final_answers.append(entity)

        answers.append((entity, similarity))

    similarities = [x[1] for x in answers]
    if similarities:
        average = np.average(similarities)
    else:
        average = 1.0

    for item in answers:
        if item[1] > average:
            final_answers.append(item[0])

    return final_answers


def main():
    model = gensim.models.Word2Vec.load("w2v_model/word2vec模型.model")  # 加载
    id_question = get_id_question()
    vocab = get_vocab()

    fout = open("data/new_result.json", "w+")

    with open("data/result_entities.json") as f:
        for index, line in enumerate(f):
            if index % 1000 == 0:
                print("正在处理line {}".format(index))
            data = json.loads(line.strip())
            question = id_question[data["question_id"]]
            candidate_entities = data["entity_answers"][0]

            final_answers = filter_entities(model, vocab, question, candidate_entities)
            data["entity_answers"] = [final_answers]
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    fout.close()


if __name__ == "__main__":
    main()
