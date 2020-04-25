.PHONY: init
init:
	poetry install --no-root

.PHONY: get
get:
	poetry run python -m news_classifier get

.PHONY: fit
fit:
	poetry run python -m news_classifier fit

.PHONY: run
run:
	poetry run streamlit run main.py
