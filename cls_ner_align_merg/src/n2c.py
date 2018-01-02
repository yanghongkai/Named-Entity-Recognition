#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Licensed under WTFPL or the Unlicense or CC0.
# This uses Python 3, but it's easy to port to Python 2 by changing
# strings to u'xx'.

import itertools
import re

def num2chinese(num, big=False, simp=True, o=False, twoalt=False):
    """
    Converts numbers to Chinese representations.
    `big`   : use financial characters.
    `simp`  : use simplified characters instead of traditional characters.
    `o`     : use 〇 for zero.
    `twoalt`: use 两/兩 for two when appropriate.
    Note that `o` and `twoalt` is ignored when `big` is used, 
    and `twoalt` is ignored when `o` is used for formal representations.
    """
    # check num first
    nd = str(num)
    if abs(float(nd)) >= 1e48:
        raise ValueError('number out of range')
    elif 'e' in nd:
        raise ValueError('scientific notation is not supported')
    c_symbol = '正负点' if simp else '正負點'
    if o:  # formal
        twoalt = False
    if big:
        c_basic = '零壹贰叁肆伍陆柒捌玖' if simp else '零壹貳參肆伍陸柒捌玖'
        c_unit1 = '拾佰仟'
        c_twoalt = '贰' if simp else '貳'
    else:
        c_basic = '〇一二三四五六七八九' if o else '零一二三四五六七八九'
        c_unit1 = '十百千'
        if twoalt:
            c_twoalt = '两' if simp else '兩'
        else:
            c_twoalt = '二'
    c_unit2 = '万亿兆京垓秭穰沟涧正载' if simp else '萬億兆京垓秭穰溝澗正載'
    revuniq = lambda l: ''.join(k for k, g in itertools.groupby(reversed(l)))
    nd = str(num)
    result = []
    if nd[0] == '+':
        result.append(c_symbol[0])
    elif nd[0] == '-':
        result.append(c_symbol[1])
    if '.' in nd:
        integer, remainder = nd.lstrip('+-').split('.')
    else:
        integer, remainder = nd.lstrip('+-'), None
    if int(integer):
        splitted = [integer[max(i - 4, 0):i]
                    for i in range(len(integer), 0, -4)]
        intresult = []
        for nu, unit in enumerate(splitted):
            # special cases
            if int(unit) == 0:  # 0000
                intresult.append(c_basic[0])
                continue
            elif nu > 0 and int(unit) == 2:  # 0002
                intresult.append(c_twoalt + c_unit2[nu - 1])
                continue
            ulist = []
            unit = unit.zfill(4)
            for nc, ch in enumerate(reversed(unit)):
                if ch == '0':
                    if ulist:  # ???0
                        ulist.append(c_basic[0])
                elif nc == 0:
                    ulist.append(c_basic[int(ch)])
                elif nc == 1 and ch == '1' and unit[1] == '0':
                    # special case for tens
                    # edit the 'elif' if you don't like
                    # 十四, 三千零十四, 三千三百一十四
                    ulist.append(c_unit1[0])
                elif nc > 1 and ch == '2':
                    ulist.append(c_twoalt + c_unit1[nc - 1])
                else:
                    ulist.append(c_basic[int(ch)] + c_unit1[nc - 1])
            ustr = revuniq(ulist)
            if nu == 0:
                intresult.append(ustr)
            else:
                intresult.append(ustr + c_unit2[nu - 1])
        result.append(revuniq(intresult).strip(c_basic[0]))
    else:
        result.append(c_basic[0])
    if remainder:
        result.append(c_symbol[2])
        result.append(''.join(c_basic[int(ch)] for ch in remainder))
    return ''.join(result)

def n2c(m):
    length = len(m.string)
    content = m.group(1)
    # print("content:{}\n".format(content))
    # print("m.start:{}\nm.end:{}".format(m.start(1), m.end(1)))
    if m.start(1) > 0 and m.string[m.start(1)-1] in ':：' or m.end(1) < length and m.string[m.end(1)] in ':：' or m.end(1)<length and m.string[m.end(1)] in '年月日点':
        return content
    else:
        return num2chinese(content)

def strn2strc(sents):
    # text = re.sub(r'(\d+)', n2c, sents)
    text = re.sub(r'(\d+)', n2c, sents)
    return text


if __name__ == "__main__":
    sents = "12月13日东苑大厦13号12:14点"
    # sents = "东苑大厦13号12:14点"
    text = strn2strc(sents)
    print("text:{}".format(text))

