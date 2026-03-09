.PHONY: lint typecheck test

lint:
	black --check adeptly tests

typecheck:
	mypy adeptly

test:
	pytest
