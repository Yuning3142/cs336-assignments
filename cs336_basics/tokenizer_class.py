import re

from cs336_basics.bpe import PAT
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
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.byte_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_match = {pair: i for i, pair in enumerate(merges)}



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

    def encode(self, text):
        chunks = chunk_token(self.special_tokens, text)

        result = []

        for chunk in chunks:
            if chunk not in self.special_tokens:
                matchs = regex.finditer(PAT, chunk)

                for match in matchs:
                    token_now = match.group().encode("utf-8")

                    token_now_id = [
                        self.byte_vocab[bytes([i])]
                        for i in token_now
                    ]

                    while True:
                        pairs = [
                            (
                                self.vocab[token_now_id[i]],
                                self.vocab[token_now_id[i + 1]]
                            )
                            for i in range(len(token_now_id) - 1)
                        ]

                        rank = float("inf")
                        index = -1

                        for i in range(len(pairs)):
                            if pairs[i] in self.merge_match:
                                if self.merge_match[pairs[i]] < rank:
                                    rank = self.merge_match[pairs[i]]
                                    index = i

                        if index == -1:
                            break

                        merge_pair = pairs[index][0] + pairs[index][1]

                        token_now_id[index] = self.byte_vocab[merge_pair]
                        del token_now_id[index + 1]

                    result += token_now_id

            else:
                result.append(
                    self.byte_vocab[chunk.encode("utf-8")]
                )

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
# print(ids)
# assert tokenizer.decode(ids) == text
#
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
#
