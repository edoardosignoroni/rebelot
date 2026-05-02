PYTHON				:= venv/bin/python3
LANGS				:= ITA ENG
FILTER_SCRIPT		:= filter_subtitles.py
PREPROCESS_SCRIPT	:= preprocess_data.py

SPLITS				:= train valid test all
JSONL_DIR			:= monolingual

RAW_DATA_DIR		:= raw_data
ALL_SIZE			:= 11186
TRAIN_SIZE			:= 8950
VALID_SIZE			:= 1118
TEST_SIZE			:= 1118

raw_data/%.txt: raw_data/%_SUBTITLES.txt
	@echo "Filtering subtitles for $*"
	$(PYTHON) $(FILTER_SCRIPT) $< > $@

filter: $(FILTERED_TARGETS)
	@echo "All subtitle files filtered."

FILTERED_TARGETS	:= \
	$(foreach lang,$(LANGS),raw_data/$(lang).txt)

FILES_SPLIT := \
	$(foreach lang,$(LANGS), \
		$(foreach split,$(SPLITS), \
			monolingual/$(lang)/$(split).jsonl))

$(JSONL_DIR)/%/all.jsonl: | $(RAW_DATA_DIR)/%.txt
	@echo "Preprocessing all data for $*"
	@mkdir -p $(@D)
	@head -n $(ALL_SIZE) $(RAW_DATA_DIR)/$*.txt > temp
	$(PYTHON) $(PREPROCESS_SCRIPT) temp $* > $@
	@rm temp

$(JSONL_DIR)/%/train.jsonl: | $(JSONL_DIR)/%/all.jsonl
	@echo "Creating train split for $*"
	@mkdir -p $(@D)
	@head -n $(TRAIN_SIZE) $(JSONL_DIR)/$*/all.jsonl > $@

$(JSONL_DIR)/%/valid.jsonl: | $(JSONL_DIR)/%/all.jsonl
	@echo "Creating valid split for $*"
	@mkdir -p $(@D)
	@tail -n $(shell echo $$(($(VALID_SIZE)+$(TEST_SIZE)))) $(JSONL_DIR)/$*/all.jsonl | head -n $(VALID_SIZE) > $@

$(JSONL_DIR)/%/test.jsonl: | $(JSONL_DIR)/%/all.jsonl
	@echo "Creating test split for $*"
	@mkdir -p $(@D)
	@tail -n $(TEST_SIZE) $(JSONL_DIR)/$*/all.jsonl > $@

CS_JSONL_DIR 	:= code-mixing/jsonl
JSONL2HF_SCRIPT := jsonl_to_hf_dataset.py
HF_CS_DIR 		:= code-mixing/hf_datasets

hf-cs-dataset: $(foreach split,$(SPLITS),$(CS_JSONL_DIR)/$(split).jsonl)
	@mkdir -p $(HF_CS_DIR); \
	for split in $(SPLITS); do \
		echo "Converting code-mixing $$split split to Hugging Face dataset format"; \
		$(PYTHON) $(JSONL2HF_SCRIPT) $(CS_JSONL_DIR)/$$split.jsonl $(HF_CS_DIR)/code-mixing-$$split ; \
	done


.PHONY: all filter clean hf-cs-dataset

splits: $(FILES_SPLIT)
	@echo "All data splits created."

all: splits filter hf-cs-dataset
	@echo "Data preparation complete."

clean:
	rm -f $(FILES_SPLIT) $(FILTERED_TARGETS)
	rm -rf $(HF_CS_DIR)/*
	