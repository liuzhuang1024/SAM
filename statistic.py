import os
import glob
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm


statis = {}

with open('datasets/CROHME/words_dict.txt') as f:
    chars = f.readlines()
chars = {chars[i].strip(): i for i in range(len(chars))}

with open('datasets/CROHME/train_labels.txt') as f:
    lines = f.readlines()

for line in tqdm(lines):

    if r'\sqrt [' in line:
        name, *symbols = line.split()
        tmp = []
        stack = []
        for i in range(len(symbols)):
            if symbols[i] == r'\sqrt' and i + 1 < len(symbols) and symbols[i+1] == '[':
                tmp.append(r'\sqrt')
                stack.append(r'\sqrt')
            elif symbols[i] == '[' and tmp[-1] == r'\sqrt':
                continue
            elif symbols[i] == ']' and len(tmp):
                tmp.pop()
            else:
                stack.append(symbols[i])
        line = name + '\t' + ' '.join(stack)

    line = line.replace(' { ', ' ').replace(' } ', ' ').replace('^', '').replace('_', '')

    name, *symbols = line.split()
    symbols = list(set(symbols))

    for i in range(len(symbols)):
        item = symbols[i]
        if item =='\\':
            print(line)
        if item not in statis:
            statis[item] = {
                'num': 1,
                'chars': {}
            }
        else:
            statis[item]['num'] += 1

        for j in range(len(symbols)):
            if symbols[j] == item:
                continue
            if symbols[j] not in statis[item]['chars']:
                statis[item]['chars'][symbols[j]] = 1
            else:
                statis[item]['chars'][symbols[j]] += 1

matrix = np.zeros((len(chars), len(chars)))

for item in statis:
    for sym in statis[item]['chars']:
        matrix[chars[item]][chars[sym]] = statis[item]['chars'][sym]/statis[item]['num']

for i in range(len(matrix)):
    for j in range(i):
        matrix[i][j] = matrix[j][i] = (matrix[i][j] + matrix[j][i]) / 2


with open('symbol_statistic_v1.json','w') as f:
    json.dump(statis, f, ensure_ascii=False)

with open('symbol_statistic_v1.pkl', 'wb') as f:
    pkl.dump(matrix,f)
