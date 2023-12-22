from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

print(tokenizer.decode([101,  6382,  4494,  4982, 26438,   437,   146,  6716,  2271,   578,
                        2152,  4010,   102,]))
print(tokenizer.decode([101,  6382,  4494,  4982,   437,   6716,  2271,   578,
                        2152,   102,]))
