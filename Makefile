.PHONY: init
init:
	poetry install

.PHONY: download
download:
	poetry run python download.py

.PHONY: train
train:
	poetry run python train.py

.PHONY: serve
serve:
	poetry run streamlit run app.py
