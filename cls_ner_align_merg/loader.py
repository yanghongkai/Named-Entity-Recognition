import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print("line:{}".format(list(line)))
        # print("line:{}".format(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                # print("sentence:{}".format(sentence))
                # "sentence:[['我', 'O'], ['要', 'O'], ['看', 'O'], ['乌', 'B-SLOC'], ['鲁', 'I-SLOC'], ['木', 'I-SLOC'], ['齐', 'I-SLOC'], ['市', 'I-SLOC'], ['第', 'I-SLOC'], ['四', 'I-SLOC'], ['十', 'I-SLOC'], ['九', 'I-SLOC'], ['中', 'I-SLOC'], ['学', 'I-SLOC'], ['东', 'I-SLOC'], ['门', 'I-SLOC'], ['去', 'O'], ['乌', 'B-ELOC'], ['鲁', 'I-ELOC'], ['木', 'I-ELOC'], ['齐', 'I-ELOC'], ['推', 'I-ELOC'], ['拿', 'I-ELOC'], ['职', 'I-ELOC'], ['业', 'I-ELOC'], ['学', 'I-ELOC'], ['校', 'I-ELOC'], ['南', 'I-ELOC'], ['门', 'I-ELOC'], ['沿', 'O'], ['西', 'B-ROAD'], ['虹', 'I-ROAD'], ['东', 'I-ROAD'], ['路', 'I-ROAD'], ['的', 'O'], ['监', 'B-TYPE'], ['控', 'I-TYPE']]"
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
                # print("word:{}".format(word))
                # word: ['监', 'B-TYPE']
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    # print("sentences:{}".format(sentences))
    # print("sentences[0]:{}".format(sentences[0]))
    # "sentences[0]:[['我', 'O'], ['要', 'O'], ['看', 'O'], ['乌', 'B-SLOC'], ['鲁', 'I-SLOC'], ['木', 'I-SLOC'], ['齐', 'I-SLOC'], ['市', 'I-SLOC'], ['第', 'I-SLOC'], ['四', 'I-SLOC'], ['十', 'I-SLOC'], ['九', 'I-SLOC'], ['中', 'I-SLOC'], ['学', 'I-SLOC'], ['东', 'I-SLOC'], ['门', 'I-SLOC'], ['去', 'O'], ['乌', 'B-ELOC'], ['鲁', 'I-ELOC'], ['木', 'I-ELOC'], ['齐', 'I-ELOC'], ['推', 'I-ELOC'], ['拿', 'I-ELOC'], ['职', 'I-ELOC'], ['业', 'I-ELOC'], ['学', 'I-ELOC'], ['校', 'I-ELOC'], ['南', 'I-ELOC'], ['门', 'I-ELOC'], ['沿', 'O'], ['西', 'B-ROAD'], ['虹', 'I-ROAD'], ['东', 'I-ROAD'], ['路', 'I-ROAD'], ['的', 'O'], ['监', 'B-TYPE'], ['控', 'I-TYPE']]"
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    # print("tag_schema:{}".format(tag_scheme))
    # tag_schema: iobes
    for i, s in enumerate(sentences):
        # print("i:{}\ts:{}".format(i,s))
        tags = [w[-1] for w in s]
        # print("tags:{}".format(tags))
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            # print("tags:{}".format(tags))
            new_tags = iob_iobes(tags)
            # print("nwe tags:{}".format(new_tags))
            for word, new_tag in zip(s, new_tags):
                # print("word:{}".format(word))
                word[-1] = new_tag
            # exit()
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    # print("lower:{}".format(lower))
    # lower: False
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # print("in char_mapping chars:{}".format(chars))
    # print("chars[0]:{}".format(chars[0]))
    # "chars[0]:['我', '要', '看', '乌', '鲁', '木', '齐', '市', '第', '四', '十', '九', '中', '学', '东', '门', '去', '乌', '鲁', '木', '齐', '推', '拿', '职', '业', '学', '校', '南', '门', '沿', '西', '虹', '东', '路', '的', '监', '控']"
    dico = create_dico(chars)
    # print("dico:{}".format(dico))
    # dico: {'仓': 16, '背': 5, '视': 348, '煨': 1, '代': 25, '欢': 2, '配': 2, '核': 5, '还': 3, '结': 4, '工': 124 }
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    # id_to_item: {0: '<PAD>', 1: '<UNK>', 2: '1', 3: ':', 4: '2', 5: '门', 6: '的', 7: '0', 8: '月'
    # item_to_id: {'俊': 402, '是': 428, '仪': 642, '哥': 728, '童': 366, '3': 12, '界': 450, '税': 876}
    # print("char_to_id:{}".format(char_to_id))
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    # print("tags:{}".format(tags))
    # "[['O', 'O', 'B-SLOC', 'I-SLOC', 'I-SLOC', 'I-SLOC', 'I-SLOC', 'I-SLOC', 'I-SLOC', 'I-SLOC', 'E-SLOC', 'O', 'B-TYPE', 'E-TYPE']]"
    dico = create_dico(tags)
    # print("dico:{}".format(dico))
    # "dico:{'E-YEAR': 1460, 'I-SPEED': 1639, 'B-SLOC': 3025, 'S-ELOC': 1, 'E-DAY': 1936, 'O': 15082, 'B-YEAR': 1460, 'B-DAY': 1936, 'I-STIME': 4285, 'I-DAY': 1367, 'I-MONTH': 478, 'E-PART': 1936, 'I-ROAD': 901, 'E-STIME': 1934, 'B-STIME': 1934, 'B-ROAD': 556, 'B-TYPE': 3027, 'I-SLOC': 19315, 'I-ETIME': 3113, 'E-ROAD': 556, 'I-YEAR': 1434, 'B-ETIME': 1414, 'E-MONTH': 1935, 'E-TYPE': 3027, 'I-ELOC': 5125, 'E-SLOC': 3025, 'B-PART': 1936, 'S-SLOC': 1, 'B-MONTH': 1935, 'E-SPEED': 1278, 'B-SPEED': 1278, 'B-ELOC': 784, 'E-ELOC': 784, 'E-ETIME': 1414}"
    tag_to_id, id_to_tag = create_mapping(dico)
    # print("tag_to_id:{}".format(tag_to_id))
    # "tag_to_id:{'I-SPEED': 17, 'E-ROAD': 30, 'B-SLOC': 7, 'E-SLOC': 8, 'B-ETIME': 21, 'I-DAY': 23, 'E-SPEED': 25, 'B-ELOC': 27, 'B-YEAR': 18, 'I-ETIME': 4, 'I-SLOC': 0, 'E-ETIME': 22, 'B-TYPE': 5, 'I-ELOC': 2, 'E-DAY': 11, 'E-MONTH': 14, 'B-PART': 10, 'B-SPEED': 24, 'E-TYPE': 6, 'E-ELOC': 28, 'O': 1, 'I-ROAD': 26, 'B-ROAD': 29, 'S-SLOC': 33, 'B-STIME': 15, 'E-PART': 12, 'E-YEAR': 19, 'S-ELOC': 32, 'B-MONTH': 13, 'B-DAY': 9, 'I-YEAR': 20, 'I-MONTH': 31, 'E-STIME': 16, 'I-STIME': 3}"
    # print("id_to_tag:{}".format(id_to_tag))
    # "id_to_tag:{0: 'I-SLOC', 1: 'O', 2: 'I-ELOC', 3: 'I-STIME', 4: 'I-ETIME', 5: 'B-TYPE', 6: 'E-TYPE', 7: 'B-SLOC', 8: 'E-SLOC', 9: 'B-DAY', 10: 'B-PART', 11: 'E-DAY', 12: 'E-PART', 13: 'B-MONTH', 14: 'E-MONTH', 15: 'B-STIME', 16: 'E-STIME', 17: 'I-SPEED', 18: 'B-YEAR', 19: 'E-YEAR', 20: 'I-YEAR', 21: 'B-ETIME', 22: 'E-ETIME', 23: 'I-DAY', 24: 'B-SPEED', 25: 'E-SPEED', 26: 'I-ROAD', 27: 'B-ELOC', 28: 'E-ELOC', 29: 'B-ROAD', 30: 'E-ROAD', 31: 'I-MONTH', 32: 'S-ELOC', 33: 'S-SLOC'} "
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        try:
            # print("s:{}".format(s))
            # "s:[['我', 'O'], ['要', 'O'], ['看', 'O'], ['乌', 'B-SLOC'], ['鲁', 'I-SLOC'], ['木', 'I-SLOC'], ['齐', 'I-SLOC'], ['市', 'I-SLOC'], ['第', 'I-SLOC'], ['四', 'I-SLOC'], ['十', 'I-SLOC'], ['九', 'I-SLOC'], ['中', 'I-SLOC'], ['学', 'I-SLOC'], ['东', 'I-SLOC'], ['门', 'E-SLOC'], ['去', 'O'], ['乌', 'B-ELOC'], ['鲁', 'I-ELOC'], ['木', 'I-ELOC'], ['齐', 'I-ELOC'], ['推', 'I-ELOC'], ['拿', 'I-ELOC'], ['职', 'I-ELOC'], ['业', 'I-ELOC'], ['学', 'I-ELOC'], ['校', 'I-ELOC'], ['南', 'I-ELOC'], ['门', 'E-ELOC'], ['沿', 'O'], ['西', 'B-ROAD'], ['虹', 'I-ROAD'], ['东', 'I-ROAD'], ['路', 'E-ROAD'], ['的', 'O'], ['监', 'B-TYPE'], ['控', 'E-TYPE']]"
            string = [w[0] for w in s]
            # print("string:{}".format(string))
            # "string:['我', '要', '看', '乌', '鲁', '木', '齐', '市', '第', '四', '十', '九', '中', '学', '东', '门', '去', '乌', '鲁', '木', '齐', '推', '拿', '职', '业', '学', '校', '南', '门', '沿', '西', '虹', '东', '路', '的', '监', '控']"
            chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                     for w in string]
            # print("chars:{}".format(chars))
            # "chars:[15, 53, 26, 52, 54, 48, 51, 58, 72, 108, 74, 173, 42, 46, 32, 5, 44, 52, 54, 48, 51, 526, 525, 197, 100, 46, 85, 31, 5, 87, 39, 782, 32, 43, 6, 62, 61]"
            segs = get_seg_features("".join(string))
            # print("segs:{}".format(segs))
            # segs: [1, 3, 0, 1, 2, 2, 2, 3, 1, 2, 2, 3, 1, 3, 1, 3, 0, 1, 2, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3, 1, 3,
                   # 0, 1, 3]
            if train:
                tags = [tag_to_id[w[-1]] for w in s]
            else:
                tags = [none_index for _ in chars]
            # print("string:{}\nchars:{}\nsegs:{}\ntags:{}\n".format(string, chars, segs, tags))
            # "string:['我', '要', '看', '乌', '鲁', '木', '齐', '市', '第', '四', '十', '九', '中', '学', '东', '门', '去', '乌', '鲁', '木', '齐', '推', '拿', '职', '业', '学', '校', '南', '门', '沿', '西', '虹', '东', '路', '的', '监', '控']
            # chars:[15, 53, 26, 52, 54, 48, 51, 58, 72, 108, 74, 173, 42, 46, 32, 5, 44, 52, 54, 48, 51, 526, 525, 197, 100, 46, 85, 31, 5, 87, 39, 782, 32, 43, 6, 62, 61]
            # segs:[1, 3, 0, 1, 2, 2, 2, 3, 1, 2, 2, 3, 1, 3, 1, 3, 0, 1, 2, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3, 1, 3, 0, 1, 3]
            # tags:[1, 1, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 27, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 28, 1, 29, 26, 26, 30, 1, 5, 6]"exit()
            data.append([string, chars, segs, tags])
        except:
            continue

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                # print("char not in dictionary:{}".format(char))
                dictionary[char] = 0
    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

