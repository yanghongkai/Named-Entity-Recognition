#!/usr/bin/env python
# coding=utf-8
import json
import os
# import pypinyin
import re
import random
import logging
from src.n2c import strn2strc


this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, '..', "data")


def del_punct(str):
    s = re.sub(r'[，。？,]', '', str)
    return s


def del_zero_ahenum(str):
    s = re.sub(r'(?<![\d : ；])(0)(?=[1-9])', '', str)
    return s


def del_loc_bracket(str):
    s = re.sub(r'[（ ）( ) · .]', '', str)
    return s

def del_dzero(str):
    s = re.sub(r'(00)(?=[: ：])', '0', str)
    return s

def pro_speed(str):
    s = re.sub(r'付', '-', str)
    s = re.sub(r'(?<=-)(一)', '1', s)
    s = re.sub(r'(?<=-)(二)', '2', s)
    s = re.sub(r'(?<=-)(三)', '3', s)
    s = re.sub(r'(?<=-)(四)', '4', s)
    s = re.sub(r'(?<=-)(五)', '5', s)
    s = re.sub(r'(?<=-)(六)', '6', s)
    s = re.sub(r'(?<=-)(七)', '7', s)
    s = re.sub(r'(?<=-)(八)', '8', s)
    s = re.sub(r'(?<=-)(九)', '9', s)

    s = re.sub(r'(一)(?=倍)', '1', s)
    s = re.sub(r'(二)(?=倍)', '2', s)
    s = re.sub(r'(三)(?=倍)', '3', s)
    s = re.sub(r'(四)(?=倍)', '4', s)
    s = re.sub(r'(五)(?=倍)', '5', s)
    s = re.sub(r'(六)(?=倍)', '6', s)
    s = re.sub(r'(七)(?=倍)', '7', s)
    s = re.sub(r'(八)(?=倍)', '8', s)
    s = re.sub(r'(九)(?=倍)', '9', s)

    return s


def dian2num(s):
    s = re.sub(r'(一)(?=点)', '1', s)
    s = re.sub(r'(二)(?=点)', '2', s)
    s = re.sub(r'(两)(?=点)', '2', s)
    s = re.sub(r'(三)(?=点)', '3', s)
    s = re.sub(r'(四)(?=点)', '4', s)
    s = re.sub(r'(五)(?=点)', '5', s)
    s = re.sub(r'(六)(?=点)', '6', s)
    s = re.sub(r'(七)(?=点)', '7', s)
    s = re.sub(r'(八)(?=点)', '8', s)
    s = re.sub(r'(九)(?=点)', '9', s)
    s = re.sub(r'(?<=\d)(点)', ':00', s)
    return s


def process_label(reflabel):
    ref = del_zero_ahenum(reflabel)
    ref = del_loc_bracket(ref)
    ref = del_dzero(ref)
    return ref


def insert_str(str, index, substr):
    return str[:index] + substr + str[index:]


def insert_lbl(str, substr, label, asr):
    pad = " "
    sidx = str.find(substr)
    eidx = sidx + len(substr)
    # 先从后边插入，否则idx会变
    str = insert_str(str, eidx, label)
    asr = insert_str(asr, eidx, label)
    # print(str)
    str = insert_str(str, sidx, pad)
    asr = insert_str(asr, sidx, pad)
    # print(str)
    return str, asr


def label_line(line, asr, record):
    year = record['year']
    month = record['month']
    day = record['day']
    stime = record['stime']
    etime = record['etime']
    sloc = record['sloc']
    eloc = record['eloc']
    sdoor = record['sdoor']
    edoor = record['edoor']

    part = record['part']
    type = record['type']
    speed = record['speed']
    road = record['road']

    year_lbl = "/YEAR "
    month_lbl = "/MONTH "
    day_lbl = "/DAY "
    stime_lbl = "/STIME "
    etime_lbl = "/ETIME "
    sloc_lbl = "/SLOC "
    eloc_lbl = "/ELOC "
    sdoor_lbl = "/SDOOR "
    edoor_lbl = "/EDOOR "
    part_lbl = "/PART "
    type_lbl = "/TYPE "
    speed_lbl = "/SPEED "
    road_lbl = "/ROAD "


    if year:
        year = process_label(year)
        print("year:{}".format(year))
        line, asr = insert_lbl(line, year, year_lbl, asr)
    if month:
        month = process_label(month)
        print("month:{}".format(month))
        line, asr = insert_lbl(line, month, month_lbl, asr)
    if day:
        day = process_label(day)
        print("day:{}".format(day))
        line, asr = insert_lbl(line, day, day_lbl, asr)

    if stime:
        stime = process_label(stime)
        print("stime:{}".format(stime))
        line, asr = insert_lbl(line, stime, stime_lbl, asr)

    if etime:
        etime = process_label(etime)
        print("etime:{}".format(etime))
        line, asr = insert_lbl(line, etime, etime_lbl, asr)

    if sloc:
        sloc = process_label(sloc)
        print("sloc:{}".format(sloc))
        line, asr = insert_lbl(line, sloc, sloc_lbl, asr)

    if eloc:
        eloc = process_label(eloc)
        print("eloc:{}".format(eloc))
        line, asr = insert_lbl(line, eloc, eloc_lbl, asr)

    if sdoor:
        sdoor = process_label(sdoor)
        print("sdoor:{}".format(sdoor))
        line, asr = insert_lbl(line, sdoor, sdoor_lbl, asr)

    if edoor:
        edoor = process_label(edoor)
        print("edoor:{}".format(edoor))
        line, asr = insert_lbl(line, edoor, edoor_lbl, asr)

    if part:
        part = process_label(part)
        print("part:{}".format(part))
        line, asr = insert_lbl(line, part, part_lbl, asr)

    if type:
        type = process_label(type)
        print("type:{}".format(type))
        line, asr = insert_lbl(line, type, type_lbl, asr)

    if speed:
        speed = process_label(speed)
        print("speed:{}".format(speed))
        line, asr = insert_lbl(line, speed, speed_lbl, asr)

    if road:
        road = process_label(road)
        print("road:{}".format(road))
        line, asr = insert_lbl(line, road, road_lbl, asr)

    line = re.sub(r'\s', ' ', line)
    asr = re.sub(r'\s', ' ', asr)
    return line, asr


def ref_label(asrpath, ralignpath, ealignpath, refpath, rlinepath, alinepath):
    data_path = os.path.join(data_dir, "new_label_1213.json")
    asrout = open(asrpath, "w", encoding="utf-8")
    refout = open(refpath, "w", encoding="utf-8")
    ralignout = open(ralignpath, "w", encoding="utf-8")
    ealignout = open(ealignpath, "w", encoding="utf-8")
    rline_out = open(rlinepath, "w", encoding="utf-8")
    aline_out = open(alinepath, "w", encoding="utf-8")
    with open(data_path, encoding="utf-8") as f:
        l_label = json.load(f)
    num = 0
    right = 0
    for counter,record in enumerate(l_label):
        # print("record:{}".format(record))
        ref = record["ref"]
        ref = del_punct(ref)
        ref = del_zero_ahenum(ref)
        ref = del_loc_bracket(ref)
        ref = del_dzero(ref)
        asr = record["asr"]
        asr = del_punct(asr)
        asr = strn2strc(asr)
        asr = pro_speed(asr)
        asr = dian2num(asr)
        if len(ref) != len(asr):
            # eout.write("ref:{}\nasr:{}\n".format(ref, asr))
            ealignout.write("ref:{}\nasr:{}\n".format(ref, asr))
            num+=1

        if len(ref) == len(asr):
            right +=1
            print("right line:{}".format(right))

            # rout.write("ref:{}\nasr:{}\n".format(ref, asr))
            ralignout.write("ref:{}\nasr:{}\n".format(ref, asr))
            line_lbl, asr_lbl = label_line(ref, asr, record)
            # print("line_lbl:{}\nasr_lbl:{}".format(line_lbl, asr_lbl))
            # ralignout.write("line_lbl:{}\nasr_lbl:{}\n".format(line_lbl, asr_lbl))
            # if right == 386:
            #     print("line_lbl:{}\nasr_lbl:{}".format(line_lbl, asr_lbl))
            #     exit()
            asrout.write("{}\n".format(asr_lbl))
            refout.write("{}\n".format(line_lbl))
            rline_out.write("{}\n".format(record['ref']))
            aline_out.write("{}\n".format(record['asr']))

        #
        # if counter>=2:
        #     break

    print("not equal line count:{}".format(num))


def gen_wl(w, l="O"):
    str = ""
    for idx,item in enumerate(w):
        if idx ==0:
            if l == "O":
                str += item + " " + l + "\n"
            else:
                str += item + " B-" + l +"\n"
        else:
            if l == "O":
                str += item + " " + l + "\n"
            else:
                str += item + " I-" + l+"\n"
    return str


def generate_train(asrpath, trainpath, devpath, testpath, log_path):
    train = open(trainpath, "w", encoding="utf-8")
    dev = open(devpath, "w", encoding="utf-8")
    test = open(testpath, "w", encoding="utf-8")
    logger = get_logger(log_path)
    num = 0
    train_cnt = 0
    dev_cnt = 0
    test_cnt = 0
    random.seed(1000)
    with open(asrpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num +=1
            # print("line:{}".format(num))
            arr = line.split()
            # print("arr:{}".format(arr))
            new_line = ""
            try:
                for item in arr:
                    wl = item.split('/')
                    if len(wl)>1:
                        w, l = wl
                        new_line += gen_wl(w, l)
                    else:
                        w = wl[0]
                        new_line += gen_wl(w)
                rv = random.random()
                # print("rv:{}\n".format(rv))
                if 0 <= rv < 0.1:
                    dev.write("{}\n".format(new_line))
                    dev_cnt +=1
                elif 0.1 <= rv < 0.2:
                    test.write("{}\n".format(new_line))
                    test_cnt +=1
                else:
                    train.write("{}\n".format(new_line))
                    train_cnt += 1
            except:
                print("error line:{}".format(num))
                continue
    logger.info("train count:{}\tdev count:{}\ttest count:{}".format(train_cnt, dev_cnt, test_cnt))
    train.close()
    dev.close()
    test.close()


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


if __name__ == "__main__":
    # ref_label
    asrpath = os.path.join(data_dir, "align", "asr_lbl.txt")
    print("asrpath:{}".format(asrpath))
    ralignpath = os.path.join(data_dir, "align", "r_align.txt")
    ealignpath = os.path.join(data_dir, "align", "e_align.txt")
    refpath = os.path.join(data_dir, "align", "ref_lbl.txt")
    epath = os.path.join(data_dir, "align", "not_align.txt")
    rlinepath = os.path.join(data_dir, "align", "rline.txt")
    alinepath = os.path.join(data_dir, "align", "aline.txt")
    ref_label(asrpath, ralignpath, ealignpath, refpath, rlinepath, alinepath)
    # ref_label

    asrpath = os.path.join(data_dir, "align", "asr_lbl.txt")
    trainpath = os.path.join(data_dir, "data_w", "train.txt")
    devpath = os.path.join(data_dir, "data_w", "dev.txt")
    testpath = os.path.join(data_dir, "data_w", "test.txt")
    logpath = os.path.join(data_dir, "data_w", "data.log")
    generate_train(asrpath, trainpath, devpath, testpath, logpath)



