#!/usr/bin/env python
# coding=utf-8
import json
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, '..', "data")


def convert_none(path, npath, asr_path, ref_path):
    """
    output new_label.json
    :param path:
    :param npath:
    :param asr_path:
    :param ref_path:
    :return:
    """
    with open(path, encoding="utf-8") as f:
        l_label = json.load(f)
    for record in l_label:
        record.update((k, "") for k, v in record.items() if v == "None")

    index = 0
    with open(asr_path, "r", encoding="utf-8") as fa, open(ref_path, "r", encoding="utf-8") as fr:
        for la in fa:
            lr = fr.readline()
            if not la or not lr:
                print("error align")
                break
            l_label[index]['ref'] = lr.strip()
            l_label[index]['asr'] = la.strip()
            index += 1

    with open(npath, "w", encoding="utf-8") as f:
        json.dump(l_label, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    path = os.path.join(data_dir, "label_1213.json")
    npath = os.path.join(data_dir, "new_label_1213.json")
    asr_path = os.path.join(data_dir, "asr_1_4000.txt")
    ref_path = os.path.join(data_dir, "data_1_4000.txt")
    convert_none(path, npath, asr_path, ref_path)


