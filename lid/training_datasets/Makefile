PYTHON				:= venv/bin/python
SCRIPT_LMO_SNIPPETS := dataset_lombard_snippets.py

SNIPPETS_COUNT 		:= 1
SNIPPETS_LENGTH 	:= 10

PATH_TO_DATA 		:= ../data/monolingual
SYNTHETIC_NAME		:= lmo-LMO-ITA-ENG-0.1-random-S0$(SNIPPETS_COUNT)-L$(SNIPPETS_LENGTH)-42

DATASET_DICT_SCRIPT := dataset_dict.py
CS_DATASETS_PATH 	:= ../data/code-mixing/hf_datasets/code-mixing-
SPLITS 				:= train valid test

MERGE_SCRIPT 		:= merge_datasets.py
CS_DATASET_PATHS 	:= $(addprefix $(CS_DATASETS_PATH),$(SPLITS))

DATASETS_DIR		:= hf_datasets
STAMP_DIR			:= stamps

$(STAMP_DIR)/synthetic:
	mkdir -p $(STAMP_DIR)
	mkdir -p $(DATASETS_DIR)
	$(PYTHON) $(SCRIPT_LMO_SNIPPETS) \
		0.1 LMO-ITA-ENG $(PATH_TO_DATA) \
		$(SNIPPETS_COUNT) $(SNIPPETS_LENGTH) \
		$(DATASETS_DIR)/$(SYNTHETIC_NAME)
	@touch $@;

$(STAMP_DIR)/code-mixing:
	mkdir -p $(STAMP_DIR)
	mkdir -p $(DATASETS_DIR)
	$(PYTHON) $(DATASET_DICT_SCRIPT) $(CS_DATASET_PATHS) $(DATASETS_DIR)/code-mixing
	@touch $@;

$(STAMP_DIR)/synthetic-cm:
	mkdir -p $(STAMP_DIR)
	$(PYTHON) $(MERGE_SCRIPT) $(DATASETS_DIR)/code-mixing $(DATASETS_DIR)/$(SYNTHETIC_NAME) $(DATASETS_DIR)/synthetic-cm
	@touch $@;

synthetic: $(STAMP_DIR)/synthetic

code-mixing: $(STAMP_DIR)/code-mixing

synthetic-cm: $(STAMP_DIR)/synthetic $(STAMP_DIR)/code-mixing $(STAMP_DIR)/synthetic-cm

all: synthetic-cm synthetic code-mixing

clean:
	rm -r $(DATASETS_DIR)
	rm -r stamps

.PHONY: all synthetic-cm synthetic code-mixing clean

