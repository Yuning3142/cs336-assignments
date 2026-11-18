import re

from .bpe import PAT
import regex
import pickle


def chunk_token(special_tokens, text):
    len_spec = len(special_tokens)
    if len_spec == 0:
        return [text]

    special_tokens.sort(key=lambda x: len(x), reverse=True)
    pattern = "("
    for i in range(len_spec):
        pattern += re.escape(special_tokens[i])
        if i < len_spec - 1:
            pattern += "|"
    pattern += ")"

    chunks = re.split(pattern, text)
    return chunks

class Tokenizer:
    def __init__(self,vocab,merges,special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.byte_vocab = {}
        for k,v in self.vocab.items():
            self.byte_vocab[v] = k



    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_iterable(self,iterable):

        for i in iterable:
            yield from self.encode(i)

    def encode(self,text):
        chunks = chunk_token(self.special_tokens, text)

        result = []
        for chunk in chunks:
            if chunk not in self.special_tokens:
                matchs = regex.finditer(PAT,chunk)

                for match in matchs:
                    token_now = match.group().encode('utf-8')
                    token_now_id = []
                    for i in token_now:
                        token_now_id.append(self.byte_vocab[bytes([i])])
                    for a,b in self.merges:
                        merge = a + b
                        if merge in token_now:
                            now = 0
                            index = 0
                            len_token  = len(token_now_id)
                            while index < len_token:
                                if index < len_token - 1:
                                    temp_token = self.vocab[token_now_id[index]] + self.vocab[token_now_id[index + 1]]

                                    if temp_token == merge:
                                        token_now_id[now] = self.byte_vocab[temp_token]
                                        now += 1
                                        index += 2
                                        continue

                                token_now_id[now] = self.byte_vocab[self.vocab[token_now_id[index]]]
                                now += 1
                                index += 1

                            token_now_id = token_now_id[:now]
                        else:
                            continue
                    result += token_now_id
            else:
                result.append(self.byte_vocab[chunk.encode('utf-8')])
        return result



    def decode(self,token_list):
        result = bytearray()
        for token in token_list:
            if token in self.vocab:
                result += self.vocab[token]
            else:
                return
        return result.decode('utf-8', errors="replace")

# vocab = {i: bytes([i]) for i in range(256)}
#
#
# special_tokens = []
# vocab[256] = b"ab"
# vocab[257] = b"abc"
# vocab[258] = b"he"
# vocab[259] = b"the"
# vocab[260] = b" world"
#
# merges = [
#     (b"a", b"b"),      # -> ab
#     (b"ab", b"c"),     # -> abc
#     (b"h", b"e"),      # -> he
#     (b"t", b"he"),     # -> the
#     (b" ", b"world"),  # 这个只是示意，实际要看你的 merge 表
# ]
#
# tokenizer = Tokenizer(vocab, merges, special_tokens)
# text = "hello world"
# ids = tokenizer.encode(text)
# assert tokenizer.decode(ids) == text
#
# texts = [
#     "hello world",
#     "abc",
# ]
#
# result = list(tokenizer.encode_iterable(texts))
# print(result)
#
# expected = []
#
# for text in texts:
#     expected.extend(tokenizer.encode(text))
#
# print(expected)
# print(result == expected)

