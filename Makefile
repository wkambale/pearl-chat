PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest

.PHONY: help setup tokenizer dataset train train-live chat test clean

help:
	@echo "Available targets:"
	@echo "  setup       install dependencies in the virtual environment"
	@echo "  tokenizer   train the Luganda byte-level BPE tokenizer"
	@echo "  dataset     prepare and tokenize the Luganda dataset"
	@echo "  train       train Pearl-Chat model in full mode"
	@echo "  train-live  run live training demo from warm-start checkpoint"
	@echo "  chat        launch the chat demonstration interface"
	@echo "  test        run the test suite"
	@echo "  clean       remove build artifacts and caches"

setup:
	uv pip install --python $(PYTHON) -e .

tokenizer:
	$(PYTHON) scripts/01_fetch_data.py
	$(PYTHON) scripts/02_train_tokenizer.py

dataset:
	$(PYTHON) scripts/03_prepare_dataset.py

train:
	$(PYTHON) scripts/04_train_pearlchat.py --mode full

train-live:
	$(PYTHON) scripts/04_train_pearlchat.py --mode live

chat:
	$(PYTHON) ui/chat_app.py

test:
	$(PYTEST) -v tests

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
