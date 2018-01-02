# !/usr/bin/env python
# coding=utf-8
import pypinyin
import pickle as pkl
import os
import numpy as np
import operator

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from tensorflow.python.framework import tensor_util
from tensorflow.contrib.crf import viterbi_decode

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = this_dir
print("data_dir:{}".format(data_dir))


class Ner():

    def __init__(self, model_name, signature_name):
        self.model_name = model_name
        self.signature_name = signature_name
        self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = self.load_dict()
        self.num_tags = len(self.tag_to_id)

    def predict(self, sents):
        _, self.sents, self.segs, self.tags= self.sents2id(sents)
        hostport = '192.168.31.186:6000'

        host, port = hostport.split(':')
        channel = implementations.insecure_channel(host, int(port))

        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        # build request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['input_w'].CopyFrom(tf.contrib.util.make_tensor_proto(self.sents, dtype=tf.int32))
        request.inputs['input_seg'].CopyFrom(tf.contrib.util.make_tensor_proto(self.segs, dtype=tf.int32))
        request.inputs['target'].CopyFrom(tf.contrib.util.make_tensor_proto(self.tags, dtype=tf.int32))
        request.inputs['dropout'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
        model_results = stub.Predict(request, 60.0)

        trans = tensor_util.MakeNdarray(model_results.outputs["trans"])
        scores = tensor_util.MakeNdarray(model_results.outputs["scores"])
        lengths = tensor_util.MakeNdarray(model_results.outputs["lengths"])
        batch_paths = self.decode(scores, lengths, trans)
        tags = [self.id_to_tag[idx] for idx in batch_paths[0]]
        item = self.result_to_json(sents, tags)
        lbl_list = ["O"]*len(sents)
        for lbldict in item["entities"]:
            start, end, lbl = lbldict["start"], lbldict["end"], lbldict["type"]
            lbl_list[start:end] = [lbl]*(end-start)
        ner_str = ""
        for c,lbl in zip(sents, lbl_list):
            ner_str += c +"/" +lbl + " "
        ner_str = ner_str.rstrip()

        year_dict = {"YEAR": 1}
        year_str = self.str_spec(sents, lbl_list, year_dict)
        # print("year_str:{}".format(year_str))
        month_dict = {"MONTH": 1}
        month_str = self.str_spec(sents, lbl_list, month_dict)
        # print("month_str:{}".format(month_str))
        day_dict = {"DAY": 1}
        day_str = self.str_spec(sents, lbl_list, day_dict)
        # print("day_str:{}".format(day_str))
        part_dict = {"PART": 1}
        part_str = self.str_spec(sents, lbl_list, part_dict)
        # print("part_str:{}".format(part_str))
        speed_dict = {"SPEED": 1}
        speed_str = self.str_spec(sents, lbl_list, speed_dict)
        # print("speed_str:{}".format(speed_str))
        type_dict = {"TYPE": 1}
        type_str = self.str_spec(sents, lbl_list, type_dict)
        # print("type_str:{}".format(type_str))

        loc_dict = {"SLOC": 1, "ELOC":2}
        loc_str = self.str_spec(sents, lbl_list, loc_dict)
        # print("loc_str:{}".format(loc_str))
        time_dict = {"STIME": 1, "ETIME": 2}
        time_str = self.str_spec(sents, lbl_list, time_dict)
        # print("time_str:{}".format(time_str))

        return loc_str, time_str, year_str, month_str, day_str, part_str, speed_str, type_str

    def load_dict(self):
        map_file_path = os.path.join(data_dir, "maps.pkl")
        with open(map_file_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pkl.load(f)
        return char_to_id, id_to_char, tag_to_id, id_to_tag

    def sents2id(self, line):
        inputs = list()
        inputs.append([line])
        line.replace(" ", "$")
        inputs.append([[self.char_to_id[char] if char in self.char_to_id else self.char_to_id["<UNK>"]
                        for char in line]])
        inputs.append([[]])
        inputs.append([[]])
        return inputs

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

    def result_to_json(self, string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item

    def str_spec(self, sents, lbl_list, spec_lbl_dict):
        spec_str = ""
        for c,lbl in zip(sents, lbl_list):
            if lbl in spec_lbl_dict:
                spec_str += c + "/" + str(spec_lbl_dict[lbl]) + " "
            else:
                spec_str += c + "/" + "0" + " "
        return spec_str


if __name__ == "__main__":
    sents = "请调出汇坤园北门2016年4月9日中午11:40的录像，四倍速回放"
    ner = Ner("ner", "ner")
    ner.predict(sents)






