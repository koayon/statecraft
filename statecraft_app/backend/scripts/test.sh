#!/usr/bin/env bash

set -e
set -x

coverage run --source=app -m pytest --capture=no
coverage report --show-missing
coverage html --title "${@-coverage}"
