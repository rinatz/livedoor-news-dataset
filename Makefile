.PHONY: init
init:
	pipenv sync
	pipenv run init

.PHONY: save
save:
	pipenv run save

.PHONY: fit
fit:
	pipenv run fit

.PHONY: build
build:
	pipenv run build

.PHONY: serve
serve:
	pipenv run serve
