.PHONY: help install test download-sample download-full clean examples

PYTHON := python3
DATA_DIR := sep_data
MAX_SAMPLE := 10
MAX_FULL := 1700

help:
	@echo "SEP Alpaca LLM Preparation - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install required dependencies"
	@echo ""
	@echo "Download:"
	@echo "  make test             - Download 10 articles for testing"
	@echo "  make download-sample  - Download 50 sample articles"
	@echo "  make download-medium  - Download 200 articles"
	@echo "  make download-full    - Download all SEP articles (takes hours!)"
	@echo ""
	@echo "Usage:"
	@echo "  make examples         - Run example usage scripts"
	@echo "  make stats            - Show dataset statistics"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove downloaded data"
	@echo "  make clean-cache      - Remove only cache files"
	@echo ""
	@echo "Advanced:"
	@echo "  make download N=100   - Download specific number of articles"
	@echo ""

install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements_minimal.txt
	@echo "✓ Installation complete"

test:
	@echo "Running test download (10 articles)..."
	$(PYTHON) sep_alpaca_prep.py --max-articles $(MAX_SAMPLE) --output-dir $(DATA_DIR)_test
	@echo "✓ Test complete - check $(DATA_DIR)_test/"

download-sample:
	@echo "Downloading sample (50 articles)..."
	$(PYTHON) sep_alpaca_prep.py --max-articles 50 --output-dir $(DATA_DIR)
	@echo "✓ Sample download complete"

download-medium:
	@echo "Downloading medium corpus (200 articles)..."
	$(PYTHON) sep_alpaca_prep.py --max-articles 200 --output-dir $(DATA_DIR)
	@echo "✓ Medium corpus download complete"

download-full:
	@echo "WARNING: This will download ~1,700 articles and take several hours!"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	$(PYTHON) sep_alpaca_prep.py --output-dir $(DATA_DIR)
	@echo "✓ Full download complete"

download:
	@if [ -z "$(N)" ]; then \
		echo "Error: Please specify N=<number>"; \
		echo "Example: make download N=100"; \
		exit 1; \
	fi
	@echo "Downloading $(N) articles..."
	$(PYTHON) sep_alpaca_prep.py --max-articles $(N) --output-dir $(DATA_DIR)

examples:
	@echo "Running example usage scripts..."
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "Error: No data found. Run 'make test' first."; \
		exit 1; \
	fi
	$(PYTHON) example_usage.py

stats:
	@echo "Dataset Statistics:"
	@echo "=================="
	@if [ -f "$(DATA_DIR)/metadata/corpus_statistics.json" ]; then \
		cat $(DATA_DIR)/metadata/corpus_statistics.json | $(PYTHON) -m json.tool; \
	else \
		echo "No statistics found. Download data first with 'make test'"; \
	fi

clean:
	@echo "Removing all downloaded data..."
	rm -rf $(DATA_DIR) $(DATA_DIR)_test
	@echo "✓ Cleanup complete"

clean-cache:
	@echo "Removing cache files..."
	rm -rf $(DATA_DIR)/metadata/sep_entries_list.json
	@echo "✓ Cache cleared"

list-topics:
	@echo "Available topics in dataset:"
	@if [ -d "$(DATA_DIR)/by_topic" ]; then \
		ls -1 $(DATA_DIR)/by_topic/; \
	else \
		echo "No topics found. Download data first."; \
	fi

check-dataset:
	@echo "Checking dataset..."
	@if [ -f "$(DATA_DIR)/alpaca_datasets/alpaca_philosophy.json" ]; then \
		echo "✓ Main dataset found"; \
		LINES=$$(cat $(DATA_DIR)/alpaca_datasets/alpaca_philosophy.json | $(PYTHON) -c "import sys, json; print(len(json.load(sys.stdin)))"); \
		echo "  Total instruction pairs: $$LINES"; \
	else \
		echo "✗ Dataset not found"; \
	fi
	@if [ -f "$(DATA_DIR)/alpaca_datasets/alpaca_philosophy.jsonl" ]; then \
		echo "✓ JSONL format found"; \
	fi
	@if [ -f "$(DATA_DIR)/alpaca_datasets/alpaca_philosophy_hf.json" ]; then \
		echo "✓ Hugging Face format found"; \
	fi

view-sample:
	@echo "Sample instruction from dataset:"
	@echo "================================"
	@if [ -f "$(DATA_DIR)/alpaca_datasets/alpaca_philosophy.json" ]; then \
		cat $(DATA_DIR)/alpaca_datasets/alpaca_philosophy.json | $(PYTHON) -c "import sys, json; data=json.load(sys.stdin); print('Instruction:', data[0]['instruction']); print('Output:', data[0]['output'][:200] + '...')"; \
	else \
		echo "No dataset found. Run 'make test' first."; \
	fi
