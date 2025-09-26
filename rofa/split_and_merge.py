import json
import os
from typing import List

def split_json(input_file: str, output_prefix: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)
    mid = n // 2
    part1 = data[:mid]
    part2 = data[mid:]

    out1 = f"{output_prefix}_part1.json"
    out2 = f"{output_prefix}_part2.json"
    with open(out1, 'w', encoding='utf-8') as f:
        json.dump(part1, f, ensure_ascii=False, indent=2)
    with open(out2, 'w', encoding='utf-8') as f:
        json.dump(part2, f, ensure_ascii=False, indent=2)


def merge_json(files: List[str], output_file: str):
    merged = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged.extend(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    split_json("coco2014_neg.json", "data")
    # merge_json(["data_part1.json", "data_part2.json"], "coco2014_neg.json")
