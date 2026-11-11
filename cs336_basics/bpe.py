import regex
from collections import Counter
import re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initilization_vocab(special_tokens):
    '''

    Args:
        special_tokens: list

    Returns:
        vocab: dict
        num: int
    '''
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(256,256 + len(special_tokens)):
        vocab[i] = special_tokens[i - 256].encode('utf-8')
    return vocab,256 + len(special_tokens)

def initilization_pre_token(input_path,special_tokens):
    pre_token = Counter()
    word_id = dict()

    with open(input_path, "r", encoding= 'utf-8') as f:
        text = f.read()
        pattern = "("
        N = len(special_tokens)
        if N != 0:
            special_tokens.sort(key = lambda x: len(x), reverse = True)
            for i in range(N):
                pattern += re.escape(special_tokens[i])
                if i != N - 1:
                    pattern += "|"
            pattern += ")"

            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk not in special_tokens:
                match = regex.finditer(PAT, chunk)

                for ele in match:
                    pre_token.update([ele.group().encode('utf-8')])
                    if ele.group().encode('utf-8') not in word_id:
                        word_id[ele.group().encode('utf-8')] = list(ele.group().encode('utf-8'))
    return pre_token, word_id


def initialization_pair_frequency(pre_token):
    pair_frequency = Counter()

    for k, v in pre_token.items():
        len_temp = len(k)
        for i in range(len_temp - 1):
            now_temp = (tuple(k[i:i+2]))
            pair_frequency[now_temp] += v
    return pair_frequency




def merge_pair(pair_frequency,vocab):
    if pair_frequency is None:
        return None
    arg = max(pair_frequency.items(), key=lambda x: (x[1], vocab[x[0][0]],vocab[x[0][1]]))[0]
    prob = max(pair_frequency.items(), key=lambda x: (x[1], x[0]))[1]
    return  arg if prob > 0 else None


def apply_merge(pair_id,pair_enc,repr_num,pair_frequency,pre_token,word_id):

    for k,v in pre_token.items():
        if pair_enc in k:
            len_k = len(word_id[k])
            index = 0
            put = 0
            while index < len_k:
                if index < len_k - 1 and tuple(word_id[k][index:index + 2]) == pair_id:



                    if index >= 1:

                        pair_frequency[(word_id[k][put - 1],word_id[k][index])] -= v
                        pair_frequency[(word_id[k][put - 1], repr_num)] += v
                    if index < len_k - 2:
                        pair_frequency[tuple(word_id[k][index + 1:index + 3])] -= v
                        pair_frequency[(repr_num,word_id[k][index + 2])] += v
                    pair_frequency[pair_id] -= v
                    word_id[k][put] = repr_num
                    index = index + 2
                    put += 1

                else:
                    word_id[k][put] = word_id[k][index]
                    index = index + 1
                    put += 1
            del word_id[k][put:]
    return pair_frequency,word_id

def train_bpe(input_path,vocab_size,special_tokens):
    vocab,repr_num = initilization_vocab(special_tokens)
    pre_token,word_id = initilization_pre_token(input_path, special_tokens)

    pair_frequency = initialization_pair_frequency(pre_token)
    merge = []
    while repr_num < vocab_size:
        merg_par = merge_pair(pair_frequency,vocab)

        if merg_par is None:
            break
        vocab[repr_num] = vocab[merg_par[0]] + vocab[merg_par[1]]
        merge.append((vocab[merg_par[0]], vocab[merg_par[1]]))
        pair_frequency,word_id= apply_merge(merg_par,vocab[repr_num],repr_num,pair_frequency,pre_token,word_id)

        repr_num += 1

    return vocab, merge






