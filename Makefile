.PHONY: format lint typecheck test docs

format:
	black adeptly tests

lint:
	black --check adeptly tests
	python -m compileall -q adeptly tests

typecheck:
	mypy adeptly tests

test:
	pytest -q

docs:
	pydocmd build
