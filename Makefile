.PHONY: init
init:
	poetry install --no-root

.PHONY: fit
fit:
	poetry run python -m news_classifier

.PHONY: run
run:
	streamlit run main.py
