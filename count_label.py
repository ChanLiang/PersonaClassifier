import sys
import json

input = sys.argv[1]

with open(input, 'r', encoding='utf-8') as r:
    data = json.load(r)

cnts = [0] * 6
for pair in data:
    label = pair[-1][0]
    cnts[label + 1] += 1

res = [e/sum(cnts) for e in cnts]
print (cnts)
print (res)