import regex
from collections import Counter
import re
import os
from typing import BinaryIO
from pathlib import Path
from multiprocessing import Pool

ROOT = Path("/Users/yangyuning/assignment1-basics")




# input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




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


def pre_tokenize(chunk,special_tokens):
    pattern = "("
    N = len(special_tokens)
    if N != 0:
        special_tokens.sort(key=lambda x: len(x), reverse=True)
        for i in range(N):
            pattern += re.escape(special_tokens[i])
            if i != N - 1:
                pattern += "|"
        pattern += ")"

        chunks = re.split(pattern, chunk)
    else:
        chunks = [chunk]

    pre_token = Counter()
    word_id = dict()

    for i in chunks:
        if i not in special_tokens:
            match = regex.finditer(PAT, i)

            for ele in match:
                match_token = ele.group().encode('utf-8')
                pre_token[match_token] += 1
                if match_token not in word_id:
                    word_id[match_token] = list(match_token)
    return pre_token,word_id

def initilization_pre_token(input_path,special_tokens):

    total_word_id = dict()
    total = Counter()
    num_processes = 4
    chunks = []
    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))


    with Pool(num_processes) as pool:
        result= pool.starmap(pre_tokenize, [(chunk,special_tokens) for chunk in chunks])
        for pre_token,word_id in result:
            total += pre_token
            total_word_id.update(word_id)

    return total, total_word_id


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










