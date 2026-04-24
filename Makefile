# Unix/macOS helpers. On Windows, use the commands in README.md manually.
.PHONY: venv install run-api run-ui

venv:
	python3.12 -m venv .venv

install: venv
	./.venv/bin/pip install -r requirements.txt

run-api:
	./.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	./.venv/bin/streamlit run streamlit_app.py
