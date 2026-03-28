import argparse
import json
import re
from typing import Iterable, List, Dict, Tuple

TAG_RE = re.compile(r"<(?P<tag>[A-Za-z0-9_]+)>(?P<content>.*?)</(?P=tag)>", re.DOTALL)
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[0-9]+|[^\w\s]", re.UNICODE)

PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", ")", "]", "}", "%"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "«", "“", "‘", "§", "«"}

def iter_records(path: str) -> Iterable[Dict]:
	# Try JSON first; fallback to JSONL
	with open(path, "r", encoding="utf-8") as f:
		raw = f.read()
	try:
		data = json.loads(raw)
		if isinstance(data, list):
			for obj in data:
				yield obj
		elif isinstance(data, dict):
			if "text" in data:
				yield data
			else:
				for obj in data.values():
					if isinstance(obj, dict) and "text" in obj:
						yield obj
	except json.JSONDecodeError:
		for line in raw.splitlines():
			line = line.strip()
			if not line or line.startswith("//"):
				continue
			yield json.loads(line)

def segment_text(text: str):
	pos = 0
	for m in TAG_RE.finditer(text):
		if m.start() > pos:
			yield text[pos:m.start()], "ita"
		yield m.group("content"), m.group("tag")
		pos = m.end()
	if pos < len(text):
		yield text[pos:], "ita"

def tokenize(text: str) -> List[str]:
	return TOKEN_RE.findall(text)

def write_vert(records: Iterable[Dict], out_path: str):
	with open(out_path, "w", encoding="utf-8") as out:
		for rec in records:
			uid = rec.get("uniq_id", "")
			domain = rec.get("domain", "")
			domain_id = rec.get("domain_id", "")
			out.write(f"# Sent: {uid}_{domain}_{domain_id}\n")
			text = rec.get("text", "")
			idx = 1
			for segment, lang in segment_text(text):
				for tok in tokenize(segment):
					out.write(f"{idx}\t{tok}\t{lang}\n")
					idx += 1
			out.write("\n")

def parse_sent_header(line: str) -> Tuple[str, str, str]:
	# line format: # Sent: uniq_id_domain_domain_id
	raw = line.replace("# Sent:", "").strip()
	parts = raw.split("_", 2)
	uid = parts[0] if len(parts) > 0 else ""
	domain = parts[1] if len(parts) > 1 else ""
	domain_id = parts[2] if len(parts) > 2 else ""
	return uid, domain, domain_id

def detokenize(tokens: List[str]) -> str:
	out = []
	for tok in tokens:
		if not out:
			out.append(tok)
			continue
		prev = out[-1]
		if tok in PUNCT_NO_SPACE_BEFORE:
			out[-1] = prev + tok
		elif prev and prev[-1] in PUNCT_NO_SPACE_AFTER:
			out[-1] = prev + tok
		else:
			out.append(" " + tok)
	return "".join(out)

def write_jsonl_from_vert(in_path: str, out_path: str):
	with open(in_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
		uid = domain = domain_id = ""
		sent_tokens: List[Tuple[str, str]] = []

		def flush():
			if not sent_tokens:
				return
			parts = []
			curr_lang = None
			curr_tokens: List[str] = []
			for tok, lang in sent_tokens:
				if curr_lang is None:
					curr_lang = lang
					curr_tokens = [tok]
				elif lang == curr_lang:
					curr_tokens.append(tok)
				else:
					text = detokenize(curr_tokens)
					if curr_lang != "ita":
						text = f"<{curr_lang}>{text}</{curr_lang}>"
					parts.append(text)
					curr_lang = lang
					curr_tokens = [tok]
			if curr_tokens:
				text = detokenize(curr_tokens)
				if curr_lang != "ita":
					text = f"<{curr_lang}>{text}</{curr_lang}>"
				parts.append(text)
			full_text = "".join(parts)
			obj = {"uniq_id": uid, "domain": domain, "domain_id": domain_id, "text": full_text}
			out.write(json.dumps(obj, ensure_ascii=False) + "\n")

		for line in f:
			line = line.rstrip("\n")
			if not line:
				flush()
				sent_tokens = []
				uid = domain = domain_id = ""
				continue
			if line.startswith("# Sent:"):
				uid, domain, domain_id = parse_sent_header(line)
				continue
			if line.startswith("#"):
				continue
			cols = line.split("\t")
			if len(cols) < 3:
				continue
			_, tok, lang = cols[0], cols[1], cols[2]
			sent_tokens.append((tok, lang))

		flush()

def main():
	ap = argparse.ArgumentParser(description="Convert between JSON/JSONL and VERT with lang tags.")
	sub = ap.add_subparsers(dest="cmd", required=True)

	ap_to_vert = sub.add_parser("to-vert", help="Convert JSON/JSONL to .vert.")
	ap_to_vert.add_argument("--input", "-i", required=True)
	ap_to_vert.add_argument("--output", "-o", required=True)

	ap_to_jsonl = sub.add_parser("to-jsonl", help="Convert .vert back to JSONL.")
	ap_to_jsonl.add_argument("--input", "-i", required=True)
	ap_to_jsonl.add_argument("--output", "-o", required=True)

	args = ap.parse_args()
	if args.cmd == "to-vert":
		write_vert(iter_records(args.input), args.output)
	elif args.cmd == "to-jsonl":
		write_jsonl_from_vert(args.input, args.output)

if __name__ == "__main__":
	main()
