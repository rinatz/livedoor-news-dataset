.PHONY: init
init:
	poetry install

.PHONY: train
train:
	poetry run python train.py

.PHONY: serve
serve:
	poetry run streamlit run app.py
