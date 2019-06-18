.PHONY: init
init:
	pipenv sync
	pipenv run init

.PHONY: create
create:
	pipenv run create

.PHONY: serve
serve:
	pipenv run serve
