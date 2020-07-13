#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety"
safety check -r requirements.txt || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "safety check failed"
  
echo "pylint"
pylint api text_recognizer training || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "pylint failed failed"
  
echo "pycodestyle"
pycodestyle api text_recognizer training || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "pydocstyle check failed"

echo "mypy"
mypy api text_recognizer training || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "mypy check failed"
  
echo "bandit"
bandit -ll -r {api,text_recognizer,training} || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "bandit check failed"

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
