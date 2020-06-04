# -------------------------------------------------------------------------------------
# Graph Structured Network for Image-Text Matching implementation based on
# https://arxiv.org/abs/2004.00277.
# "Graph Structured Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2020
# ---------------------------------------------------------------
"""Data preprocessing with semantic parsing"""

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import json
import argparse

nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')


def parse_a_sentence(sent):
    return nlp.dependency_parse(sent)


def read_files(src_path,  tar_path):
    print('src_path', src_path)
    print('tar_path', tar_path)
    data = []
    reader = open(src_path, 'r')
    # writer = open(tar_path, 'w')
    for line in tqdm(reader.readlines()):
        line = line.strip()
        parse = parse_a_sentence(line)
        # store semantically dependent word ids
        sent = [(i[1:3]) for i in parse]
        data.append(sent)

    nlp.close()
    reader.close()

    with open(tar_path, "w") as f:
        json.dump(data, f)

# python dependency_parser.py
# --src_path='/media/ubuntu/data/chunxiao/f30k_precomp/dev_caps.txt'
# --tar_path='/media/ubuntu/data/chunxiao/f30k_precomp/dev_caps.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing for text data')
    parser.add_argument(
        '--src_path', type=str, help='path of data phases to be processed', required=True)
    parser.add_argument(
        '--tar_path', type=str, help='dest path of processed data', required=True)

    opt = parser.parse_args()
    read_files(opt.src_path, opt.tar_path)

    print('Done')
