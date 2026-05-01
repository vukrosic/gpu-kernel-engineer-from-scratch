.PHONY: test bench bootstrap-results lint

test:
	pytest

bench:
	python scripts/benchmark_reference.py

bootstrap-results:
	python scripts/bootstrap_results.py

lint:
	ruff check .
