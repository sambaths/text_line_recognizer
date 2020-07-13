#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety"
pipenv run safety check -r requirements.txt -r requirements-dev.txt || FAILURE=true

echo "pylint"
pipenv run pylint api text_recognizer training || FAILURE=true

echo "pycodestyle"
pipenv run pycodestyle api text_recognizer training || FAILURE=true

echo "pydocstyle"
pipenv run pydocstyle api text_recognizer training || FAILURE=true

echo "mypy"
pipenv run mypy api text_recognizer training || FAILURE=true

echo "bandit"
pipenv run bandit -ll -r {api,text_recognizer,training} || FAILURE=true

echo "shellcheck"
pipenv run shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
