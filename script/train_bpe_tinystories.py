from cs336_basics.bpe import train_bpe
import time
import pickle
from pathlib import Path

ROOT = Path("/Users/yangyuning/assignment1-basics")


def main():
    start_time = time.time()

    input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    training_time = time.time() - start_time

    longest_token = max(vocab.values(), key=len)

    print("training time:", training_time)
    print("longest token:", longest_token)
    print("longest token length:", len(longest_token))

    with open(ROOT / "tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(ROOT / "tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    main()
