# encoding = utf8
import re
import math
import codecs
import random

import numpy as np
import jieba
jieba.initialize()


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    # print("sorted_items:{}".format(sorted_items))
    # sorted_items: [('<PAD>', 10000001), ('<UNK>', 10000000), ('1', 4540), (':', 3341), ('2', 3173), ('门', 3105),
                   # ('的', 3030), ('0', 2165), ('月', 1947) ]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    # print("id_to_item:{}".format(id_to_item))
    # id_to_item: {0: '<PAD>', 1: '<UNK>', 2: '1', 3: ':', 4: '2', 5: '门', 6: '的', 7: '0', 8: '月'
    item_to_id = {v: k for k, v in id_to_item.items()}
    # print("item_to_id:{}".format(item_to_id))
    # item_to_id: {'俊': 402, '是': 428, '仪': 642, '哥': 728, '童': 366, '3': 12, '界': 450, '税': 876}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    # print("before tags:{}".format(tags))
    for i, tag in enumerate(tags):
        # print("i:{}\ttag:{}".format(i, tag))
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
    # print("after tags:{}\n".format(tags))
    # exit()
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


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []
    # print("string:{}".format(string))
    # string: 我要看乌鲁木齐市第四十九中学东门去乌鲁木齐推拿职业学校南门沿西虹东路的监控
    # print("jieba cut:{}".format(jieba.cut(string)))
    # 'jieba cut:<generator object Tokenizer.cut at 0x7f49ae2f51a8>'
    for word in jieba.cut(string):
        # print("word:{}".format(word))
        # word: 我要
        # word: 看
        # word: 乌鲁木齐市
        # word: 第四十九
        # word: 中学
        # word: 东门
        # word: 去
        # word: 乌鲁木齐
        # word: 推拿
        # word: 职业
        # word: 学校
        # word: 南门
        # word: 沿西虹
        # word: 东路
        # word: 的
        # word: 监控
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            # print("tmp:{}".format(tmp))
            # word: 乌鲁木齐市
            # tmp: [1, 2, 2, 2, 3]
            seg_feature.extend(tmp)
    # print("seg_feature:{}".format(seg_feature))
    # seg_feature: [1, 3, 0, 1, 2, 2, 2, 3, 1, 2, 2, 3, 1, 3, 1, 3, 0, 1, 2, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3, 1, 3,
                  # 0, 1, 3]
    return seg_feature


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        # print("line:{}".format(line))
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    # print("inputs:{}".format(inputs))
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        # print("len batch_data:{}".format(len(self.batch_data)))
        # 'len batch_data:152'
        # print("batch_data[0]:{}".format(self.batch_data[0]))
        # "batch_data[0]:[" \
        # "[['调', '出', '博', '奥', '的', '预', '案', 0, 0, 0], ['调', '出', '红', '军', '的', '监', '控', 0, 0, 0], ['请', '调', '出', '西', '园', '的', '预', '案', 0, 0], ['给', '我', '一', '盒', '新', '院', '的', '监', '控', 0], ['查', '看', '和', '顺', '花', '园', '的', '预', '案', 0], ['查', '看', '建', '材', '小', '区', '的', '预', '案', 0], ['给', '我', '维', '泰', '大', '厦', '的', '视', '频', 0], ['查', '看', '华', '旗', '龙', '湾', '的', '监', '控', 0], ['调', '出', '农', '行', '小', '区', '的', '视', '频', 0], ['查', '看', '大', '盛', '国', '际', '的', '视', '频', 0], ['给', '我', '南', '湖', '明', '珠', '的', '预', '案', 0], ['请', '调', '出', '长', '富', '宫', '的', '监', '控', 0], ['调', '出', '博', '鳌', '南', '门', '的', '监', '控', 0], ['调', '出', '秦', '郡', '南', '门', '的', '监', '控', 0], ['给', '我', '灭', '火', '大', '厦', '的', '监', '控', 0], ['给', '我', '南', '湖', '大', '厦', '的', '监', '控', 0], ['查', '看', '新', '新', '家', '园', '的', '监', '控', 0], ['调', '出', '金', '星', '大', '厦', '的', '监', '控', 0], ['调', '出', '向', '阳', '幼', '儿', '园', '的', '预', '案'], ['查', '看', '康', '明', '园', '后', '门', '的', '视', '频']], " \
        # "[[27, 28, 154, 339, 6, 66, 64, 0, 0, 0], [27, 28, 117, 403, 6, 62, 61, 0, 0, 0], [25, 27, 28, 39, 41, 6, 66, 64, 0, 0], [21, 15, 90, 1023, 37, 63, 6, 62, 61, 0], [56, 26, 98, 333, 71, 41, 6, 66, 64, 0], [56, 26, 127, 318, 40, 23, 6, 66, 64, 0], [21, 15, 120, 184, 57, 80, 6, 65, 67, 0], [56, 26, 79, 495, 147, 104, 6, 62, 61, 0], [27, 28, 162, 152, 40, 23, 6, 65, 67, 0], [56, 26, 57, 179, 88, 115, 6, 65, 67, 0], [21, 15, 31, 109, 171, 273, 6, 66, 64, 0], [25, 27, 28, 186, 349, 255, 6, 62, 61, 0], [27, 28, 154, 711, 31, 5, 6, 62, 61, 0], [27, 28, 538, 297, 31, 5, 6, 62, 61, 0], [21, 15, 764, 472, 57, 80, 6, 62, 61, 0], [21, 15, 31, 109, 57, 80, 6, 62, 61, 0], [56, 26, 37, 37, 69, 41, 6, 62, 61, 0], [27, 28, 96, 183, 57, 80, 6, 62, 61, 0], [27, 28, 111, 121, 101, 93, 41, 6, 66, 64], [56, 26, 227, 171, 41, 45, 5, 6, 65, 67]], " \
        # "[[1, 3, 1, 3, 0, 1, 3, 0, 0, 0], [1, 3, 1, 3, 0, 1, 3, 0, 0, 0], [0, 1, 3, 1, 3, 0, 1, 3, 0, 0], [0, 0, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 1, 3, 1, 3, 0, 1, 3, 0], [0, 1, 3, 1, 2, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 0, 1, 2, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0], [1, 3, 1, 3, 1, 2, 3, 0, 1, 3], [1, 3, 1, 2, 3, 1, 3, 0, 1, 3]], " \
        # "[[1, 1, 7, 8, 1, 5, 6, 0, 0, 0], [1, 1, 7, 8, 1, 5, 6, 0, 0, 0], [1, 1, 1, 7, 8, 1, 5, 6, 0, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 1, 7, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 8, 1, 5, 6, 0], [1, 1, 7, 0, 0, 0, 8, 1, 5, 6], [1, 1, 7, 0, 0, 0, 8, 1, 5, 6]]]"
        self.len_data = len(self.batch_data)
        # print("len_data:{}".format(self.len_data))
        # len_data: 152

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        # print("num_batch:{}".format(num_batch))
        # num_batch: 152
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []

        max_length = max([len(sentence[0]) for sentence in data])
        # print("max_length:{}".format(max_length))
        # max_length: 10
        for line in data:
            string, char, seg, target = line
            # print("string:{}".format(string))
            # print("char:{}".format(char))
            # print("seg:{}".format(seg))
            # print("target:{}".format(target))
            # string: ['调', '出', '博', '奥', '的', '预', '案']
            # char: [27, 28, 154, 339, 6, 66, 64]
            # seg: [1, 3, 1, 3, 0, 1, 3]
            # target: [1, 1, 7, 8, 1, 5, 6]
            padding = [0] * (max_length - len(string))
            # print("padding:{}".format(padding))
            # padding: [0, 0, 0]
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
            # print("strings:{}".format(strings))
            # strings: [['调', '出', '博', '奥', '的', '预', '案', 0, 0, 0]]
            # print("strings[0]:{}".format(strings[0]))
            # print("chars[0]:{}".format(chars[0]))
            # print("segs[0]:{}".format(segs[0]))
            # print("targets[0]:{}".format(targets[0]))
            # strings[0]: ['调', '出', '博', '奥', '的', '预', '案', 0, 0, 0]
            # chars[0]: [27, 28, 154, 339, 6, 66, 64, 0, 0, 0]
            # segs[0]: [1, 3, 1, 3, 0, 1, 3, 0, 0, 0]
            # targets[0]: [1, 1, 7, 8, 1, 5, 6, 0, 0, 0]
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
