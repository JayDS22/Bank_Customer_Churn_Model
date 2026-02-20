.PHONY: all fairness causal bayesian test clean install

# Default: run full audit with synthetic data
all: install
	python scripts/run_full_audit.py --use-sample-data

# Individual components
fairness: install
	python scripts/run_fairness_only.py --use-sample-data

causal: install
	python scripts/run_causal_only.py --use-sample-data

# Full audit skipping expensive components
quick: install
	python scripts/run_full_audit.py --use-sample-data --skip-bayesian

# Run tests
test:
	cd $(CURDIR) && python -m pytest tests/ -v --tb=short

# Install dependencies
install:
	pip install -r requirements.txt --break-system-packages -q 2>/dev/null || pip install -r requirements.txt -q

# Clean outputs
clean:
	rm -rf outputs/plots/*.png
	rm -rf outputs/reports/*.txt
	rm -rf outputs/reports/*.json
	rm -rf outputs/models/*.pkl
	rm -f outputs/audit_log.txt

# R analysis (requires R + AER package)
r-iv:
	Rscript R/iv_analysis.R
