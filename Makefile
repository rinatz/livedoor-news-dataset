.PHONY: init
init:
	poetry install --no-root

.PHONY: dataset
dataset:
	poetry run python dataset.py

.PHONY: train
train:
	poetry run python train.py

.PHONY: serve
serve:
	poetry run streamlit run app.py
