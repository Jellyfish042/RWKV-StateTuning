from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import argparse
import json
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Configuration")
    parser.add_argument("--text_data", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

    with open(args.text_data, 'r', encoding='utf-8') as file:
        all_data = []
        for line in tqdm(file):
            json_object = json.loads(line)

            tokenized_text = tokenizer.encode(json_object['text'])
            mask = [1 for _ in range(len(tokenized_text))]

            d = {
                'text': tokenized_text,
                'mask': mask
            }
            all_data.append(d)

    with open(args.save_path, 'w', encoding='utf-8') as file:
        for item in all_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
