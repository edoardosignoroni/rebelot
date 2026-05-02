#!/nlp/projekty/langtok/langtok_env/bin/python3
import sys

in_file = sys.argv[1]

with open(in_file, 'r', encoding='utf-8') as f:
    subtitles = f.read().splitlines()

subtitles = list(filter(lambda x: len(x) > 30, subtitles))

for i, line in enumerate(subtitles):
    if line.startswith("- "):
        subtitles[i] = line[2:]
    if line.startswith("<i>") and line.endswith("</ i>"):
        subtitles[i] = line[3:-4]
    if line.startswith("\"") and line.endswith("\""):
        subtitles[i] = line[1:-1]
    alpha = 0
    other = 0
    for c in line:
        alpha += c.isalpha()
        other += not c.isalpha()
    if 2 * other > alpha:
        subtitles[i] = ''

subtitles = list(filter(lambda x: len(x) > 0, subtitles))
subtitles = list(filter(lambda x: not x.startswith('#'), subtitles))

for line in subtitles:
    sys.stdout.write(line + '\n')

