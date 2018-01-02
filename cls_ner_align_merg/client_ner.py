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
        # print("tag_to_id:{}".format(self.tag_to_id))
        # "tag_to_id:{'I-STIME': 3, 'E-SLOC': 8, 'B-MONTH': 13, 'I-SLOC': 0, 'B-SLOC': 7, 'E-ROAD': 30, 'I-DAY': 23, 'I-ETIME': 4, 'I-ELOC': 2, 'E-STIME': 16, 'E-ETIME': 22, 'I-SPEED': 17, 'B-TYPE': 5, 'E-ELOC': 28, 'I-ROAD': 26, 'E-PART': 12, 'E-SPEED': 25, 'S-ELOC': 32, 'B-ELOC': 27, 'B-DAY': 9, 'O': 1, 'B-PART': 10, 'I-MONTH': 31, 'B-STIME': 15, 'E-DAY': 11, 'B-ROAD': 29, 'E-MONTH': 14, 'B-YEAR': 18, 'E-TYPE': 6, 'B-ETIME': 21, 'B-SPEED': 24, 'E-YEAR': 19, 'I-YEAR': 20, 'S-SLOC': 33}"
        # print("len(tag_to_id):{}".format(len(self.tag_to_id)))
        # len(tag_to_id): 34
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
        # print("lengths:{}".format(lengths))
        # lengths: [33]

        # print("trans shape:{}".format(tensor_util.MakeNdarray(model_results.outputs["trans"]).shape))
        # print("scores shape:{}".format(tensor_util.MakeNdarray(model_results.outputs["scores"]).shape))
        # print("lengths shape:{}".format(tensor_util.MakeNdarray(model_results.outputs["lengths"]).shape))
        # 'trans shape:(35, 35) ' \
        # 'scores shape:(1, 33, 34) ' \
        # 'lengths shape:(1,)'
        batch_paths = self.decode(scores, lengths, trans)
        tags = [self.id_to_tag[idx] for idx in batch_paths[0]]
        item = self.result_to_json(sents, tags)
        print("item:{}".format(item))
        lbl_list = ["O"]*len(sents)
        # "lbl_list:['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']"
        print("lbl_list:{}".format(lbl_list))
        for lbldict in item["entities"]:
            start, end, lbl = lbldict["start"], lbldict["end"], lbldict["type"]
            print("start:{}\tend:{}\tlbl:{}".format(start, end, lbl))
            lbl_list[start:end] = [lbl]*(end-start)
            # 'start:3	end:8	lbl:SLOC
            # start:8	end:13	lbl:YEAR
            # start:13	end:15	lbl:MONTH
            # start:15	end:17	lbl:DAY
            # start:17	end:19	lbl:PART
            # start:19	end:24	lbl:STIME
            # start:25	end:27	lbl:TYPE
            # start:28	end:31	lbl:SPEED'
        print("lbl_list:{}".format(lbl_list))
        # "lbl_list:['O', 'O', 'O', 'SLOC', 'SLOC', 'SLOC', 'SLOC', 'SLOC', 'YEAR', 'YEAR', 'YEAR', 'YEAR', 'YEAR', 'MONTH', 'MONTH', 'DAY', 'DAY', 'PART', 'PART', 'STIME', 'STIME', 'STIME', 'STIME', 'STIME', 'O', 'TYPE', 'TYPE', 'O', 'SPEED', 'SPEED', 'SPEED', 'O', 'O']"
        ner_str = ""
        for c,lbl in zip(sents, lbl_list):
            ner_str += c +"/" +lbl + " "
        ner_str = ner_str.rstrip()
        # print("ner_str:{}".format(ner_str))

        year_dict = {"YEAR": 1}
        year_str = self.str_spec(sents, lbl_list, year_dict)
        print("year_str:{}".format(year_str))
        month_dict = {"MONTH": 1}
        month_str = self.str_spec(sents, lbl_list, month_dict)
        print("month_str:{}".format(month_str))
        day_dict = {"DAY": 1}
        day_str = self.str_spec(sents, lbl_list, day_dict)
        print("day_str:{}".format(day_str))
        part_dict = {"PART": 1}
        part_str = self.str_spec(sents, lbl_list, part_dict)
        print("part_str:{}".format(part_str))
        speed_dict = {"SPEED": 1}
        speed_str = self.str_spec(sents, lbl_list, speed_dict)
        print("speed_str:{}".format(speed_str))
        type_dict = {"TYPE": 1}
        type_str = self.str_spec(sents, lbl_list, type_dict)
        print("type_str:{}".format(type_str))

        loc_dict = {"SLOC": 1, "ELOC":2}
        loc_str = self.str_spec(sents, lbl_list, loc_dict)
        print("loc_str:{}".format(loc_str))
        time_dict = {"STIME": 1, "ETIME": 2}
        time_str = self.str_spec(sents, lbl_list, time_dict)
        print("time_str:{}".format(time_str))

        return loc_str, time_str, year_str, month_str, day_str, part_str, speed_str, type_str

    def load_dict(self):
        map_file_path = os.path.join(data_dir, "maps.pkl")
        with open(map_file_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pkl.load(f)
        return char_to_id, id_to_char, tag_to_id, id_to_tag

    def sents2id(self, line):
        # inputs = []
        # line.replace(" ", "$")
        # inputs.append([[self.char_to_id[char] if char in self.char_to_id else self.char_to_id["<UNK>"]
        #                for char in line]])
        # print("inputs:{}".format(inputs))
        # return [[in# puts]], [[[[[]]]]]
        inputs = list()
        inputs.append([line])
        line.replace(" ", "$")
        inputs.append([[self.char_to_id[char] if char in self.char_to_id else self.char_to_id["<UNK>"]
                        for char in line]])
        print("inputs:{}".format(inputs))
        inputs.append([[]])
        inputs.append([[]])
        print("inputs:{}".format(inputs))
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
        # print("item:{}".format(item))
        # "item:{'entities': [{'start': 0, 'word': '请调', 'end': 2, 'type': 'MONTH'}, {'start': 0, 'word': '出', 'end': 3, 'type': 'DAY'}, {'start': 0, 'word': '汇坤', 'end': 5, 'type': }, {'start': 0, 'word': '园北', 'end': 7, 'type': 'DAY'}, {'start': 0, 'word': '门2', 'end': 9, 'type': 'DAY'}, {'start': 13, 'word': '016年4月', 'end': 15, 'type': 'MONTH'}, {t': 13, 'word': '9', 'end': 16, 'type': 'DAY'}, {'start': 19, 'word': '日中午11', 'end': 21, 'type': 'MONTH'}, {'start': 19, 'word': ':', 'end': 22, 'type': 'DAY'}, {'start': 1ord': '40', 'end': 24, 'type': 'DAY'}, {'start': 24, 'word': '的录', 'end': 26, 'type': 'DAY'}, {'start': 26, 'word': '像,', 'end': 28, 'type': 'DAY'}], " \
        # "'string': '请调出汇坤园016年4月9日中午11:40的录像,四倍速回放'}"
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






