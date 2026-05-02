PYTHON				:= venv/bin/python
LANGS				:= ENG ITA LMO
STAMPS_DIR			:= stamps
JSONL_DATA_PATH		:= ../../data/monolingual
HF_MONOLING_SCRIPT 	:= monolingual_dataset.py

MERGE_SCRIPT 		:= merge_dataset.py
CS_DATASET			:= ../../data/code-mixing/hf_datasets/code-mixing-test

stamp-%:
	@mkdir -p $(STAMPS_DIR)
	@echo $*;
	$(PYTHON) $(HF_MONOLING_SCRIPT) $(JSONL_DATA_PATH)/$*/test.jsonl hf_datasets/$*
	@touch $(STAMPS_DIR)/$@;

STAMP_FILES	:= \
	$(foreach l,$(LANGS), \
		stamp-$(l))

monolingual: $(STAMP_FILES)

cm-monolingual: $(STAMP_FILES)
	$(PYTHON) $(MERGE_SCRIPT) $(wildcard hf_datasets/*) test-all;

all: monolingual cm-monolingual

.PHONY: monolingual cm-monolingual all
