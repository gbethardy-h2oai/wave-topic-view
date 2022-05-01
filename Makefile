setup: ## Install dependencies
	python3 -m venv venv
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/python -m pip install -r requirements.txt

.PHONY: run
run: export STEAM_URL = https://steam.cloud.h2o.ai
run: export STEAM_TOKEN = pat_f6l87hg6pqi7kur82m61x6b7ty2yl7osmkk6
run: export OKTA_ORG_URL = https://id.cloud.h2o.ai
run: export OKTA_TOKEN = 00G35DbInGiEbyAQfZaQzlI9q3NJgX9xE2T41dK-Up
run: export OKTA_CLIENT_ID = 0oa8t0ky40B0TO4eQ4h6

run: ## Run the app with active reload
	./venv/bin/wave run src/app.py

.PHONY: format
format:
	./venv/bin/isort .
	./venv/bin/black --exclude venv .

help: ## List all make tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
