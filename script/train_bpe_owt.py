from cs336_basics.tokenizer_class import Tokenizer
from cs336_basics.tokenizer_class import chunk_token
import time
from pathlib import Path
import numpy as np

ROOT = Path("/Users/yangyuning/assignment1-basics")

def  main():
    start_time = time.time()
    vocab_filepath = ROOT / "script"/"tinystories_vocab.pkl"
    merge_filepath = ROOT / "script"/"tinystories_merges.pkl"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath, merge_filepath,special_tokens)
    valid_filepath = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"







    with open(valid_filepath, "r", encoding="utf-8") as f:
        text = f.read()
        a_time = time.time()
        ids = tokenizer.encode(text)

        time_token = time.time() - start_time
        array = np.array(ids, dtype=np.uint16)
        print(len(text.encode("utf-8"))/time_token)

        array.tofile(ROOT/"tinystories_train_tokens.bin")




    documents = text.split("<|endoftext|>")
    documents = [doc for doc in documents if doc.strip()]
    documents = documents[:10]

    compute_ratio = []
    for doc in documents:
        last_time = time.time()
        len_new = len(tokenizer.encode(doc))
        len_old = len(doc.encode("utf-8"))
        compute_ratio.append(len_old / len_new)

        print(len_old, time.time()-last_time,len_old/(time.time()-last_time))


    mean_compute_ratio = sum(compute_ratio) / len(compute_ratio)

    print(mean_compute_ratio)
    print(compute_ratio)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    main()




