PYTHON 		 	:= venv/bin/python
TRAIN_SCRIPT 	:= training.py
TRAIN_DAT_PATH	:= training_datasets/hf_datasets
TRAIN_DATASETS 	:= $(notdir $(wildcard $(TRAIN_DAT_PATH)/*))

MODELS 			:= \
	jhu-clsp/mmBERT-base google-bert/bert-base-multilingual-uncased
XXX_TAGS		:= True

EVAL_SCRIPT		:= evaluation/evaluation_verbose.py
MODELS_DIR  	:= models
STAMPS_DIR		:= stamps
RESULTS_DIR		:= results

sanitize = $(subst /,-,$(1))

define LOMBARD_TRAIN
STAMP_$(1)_$(call sanitize,$(2))_$(3) := $(STAMPS_DIR)/train--$(1)--$(call sanitize,$(2))--$(3).done

$$(STAMP_$(1)_$(call sanitize,$(2))_$(3)):
	@echo "==== TRAINING $(1) / $(2) / $(3) ===="
	@mkdir -p $(MODELS_DIR) $(STAMPS_DIR) $(RESULTS_DIR)/self
	$(PYTHON) $(TRAIN_SCRIPT) \
		$(TRAIN_DAT_PATH)/$(1) \
		$(2) \
		$(3) \
		$(MODELS_DIR)/$(call sanitize,$(2))--$(1)--$(3);
	@touch $$@
	$(PYTHON) $(EVAL_SCRIPT) \
		$(MODELS_DIR)/$(call sanitize,$(2))--$(1)--$(3) \
		$(TRAIN_DAT_PATH)/$(1) $(RESULTS_DIR)/self/$(call sanitize,$(2))--$(1)--$(3).csv > $(RESULTS_DIR)/self/$(call sanitize,$(2))--$(1)--$(3).json;
endef

$(foreach d,$(TRAIN_DATASETS), \
  $(foreach m,$(MODELS), \
	$(foreach x,$(XXX_TAGS), \
    	$(eval $(call LOMBARD_TRAIN,$(d),$(m),$(x))))))

TRAIN_TARGETS := \
$(foreach d,$(TRAIN_DATASETS), \
  $(foreach m,$(MODELS), \
	$(foreach x,$(XXX_TAGS), \
    	$(STAMPS_DIR)/train--$(d)--$(call sanitize,$(m))--$(x).done)))

EVAL_DAT_PATH		:= evaluation/eval_datasets/hf_datasets
MODELS_EVAL			:= $(notdir $(wildcard $(MODELS_DIR)/*))
EVAL_DATASETS		:= test-all

define LOMBARD_TEST
RESULT_$(1)_$(2) := $(RESULTS_DIR)/$(1)/$(2).json

$$(RESULT_$(1)_$(2)):
	@echo "==== EVALUATING $(1) / $(2) ===="
	@mkdir -p $(RESULTS_DIR)/$(1)
	$(PYTHON) $(EVAL_SCRIPT) $(MODELS_DIR)/$(2) $(EVAL_DAT_PATH)/$(1) $(RESULTS_DIR)/$(1)/$(2).csv > $(RESULTS_DIR)/$(1)/$(2).json
endef

$(foreach d,$(EVAL_DATASETS), \
  $(foreach m,$(MODELS_EVAL), \
    $(eval $(call LOMBARD_TEST,$(d),$(m)))))

EVAL_TARGETS	:= \
  $(foreach d,$(EVAL_DATASETS), \
    $(foreach m,$(MODELS_EVAL), \
	  $(RESULTS_DIR)/$(d)/$(m).json))

.PHONY: train all clean

eval: $(EVAL_TARGETS)
	@echo "==== ALL EVALUATION COMPLETE ===="

train: $(TRAIN_TARGETS)
	@echo "==== ALL TRAINING COMPLETE ===="

clean:
	rm -r $(STAMPS_DIR) $(MODELS_DIR) $(RESULTS_DIR)

all: train eval
	@echo "==== ALL TASKS COMPLETE ===="
