#!/usr/bin/env python
# coding=utf-8
import os
import pypinyin

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, '..', 'data')


def convert2pinyin(wpath, pypath):
    out_py = open(pypath, "w", encoding="utf-8")
    with open(wpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                w, lbl = line.strip().split()
                w_pinyin = pypinyin.lazy_pinyin(w, 0, errors='ignore')
                if not w_pinyin:
                    w_pinyin = [w]
                # print("w:{}\tpinyin:{}".format(w, w_pinyin))
                new_line = "{} {}".format(w_pinyin[0], lbl)
            else:
                w = '\n'
                new_line = ""
            # print("{}\n".format(new_line))
            out_py.write("{}\n".format(new_line))


if __name__ == "__main__":
    wpath = os.path.join(data_dir, "data_w", "train.txt")
    pypath = os.path.join(data_dir, "data_py", "train_pinyin.txt")
    convert2pinyin(wpath, pypath)
    wpath = os.path.join(data_dir, "data_w", "dev.txt")
    pypath = os.path.join(data_dir, "data_py", "dev_pinyin.txt")
    convert2pinyin(wpath, pypath)
    wpath = os.path.join(data_dir, "data_w", "test.txt")
    pypath = os.path.join(data_dir, "data_py", "test_pinyin.txt")
    convert2pinyin(wpath, pypath)



