#!/nlp/projekty/langtok/langtok_env/bin/python3

import sys
import json

in_file = sys.argv[1]
lang = sys.argv[2]

with open(in_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

rr = []
for line in lines:
    r = {'text': line[:-1:], 'tag': lang}
    rr.append(r)

for r in rr:
    json.dump(r, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")