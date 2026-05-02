PYTHON				:= venv/bin/python3

DATASET_SCRIPT		:= eval_dataset.py
DATASET_DATA_DIR	:= ../data/monolingual
DATASET_NAME		:= eval_dataset

eval-dataset:
	@echo "Creating dataset $(DATASET_NAME)";
	$(PYTHON) $(DATASET_SCRIPT) \
		$(DATASET_DATA_DIR)/ENG/test.jsonl \
		$(DATASET_DATA_DIR)/ITA/test.jsonl \
		$(DATASET_DATA_DIR)/LMO/test.jsonl \
		$(DATASET_NAME);
	@echo "Dataset saved into $(DATASET_NAME)";

TOOLS_DIR		:= compared_tools
TOOL_SCRIPTS 	:= $(wildcard $(TOOLS_DIR)/sents_*.py)
TOOLS 			:= $(patsubst sents_%,%,$(basename $(notdir $(TOOL_SCRIPTS))))
MODELS_DIR		:= tools_models
RESULTS_DIR		:= results

sanitize   = $(subst /,-,$(1))
model_name = $(basename $(notdir $(1)))

tools: $(TOOLS)

define NL


endef

$(TOOLS):
	@mkdir -p $(RESULTS_DIR)
	@echo "Running tool: $@"
	$(foreach model,$(wildcard $(MODELS_DIR)/$@/*), \
		$(PYTHON) $(TOOLS_DIR)/sents_$@.py $(model) $(DATASET_NAME) > $(RESULTS_DIR)/$@--$(call model_name,$(model)).csv$(NL)\
		$(PYTHON) get_metrics.py $(RESULTS_DIR)/$@--$(call model_name,$(model)).csv > $(RESULTS_DIR)/$@--$(call model_name,$(model)).txt$(NL)\
	)	

clean:
	rm -r $(DATASET_NAME)
	rm -r $(RESULTS_DIR)
