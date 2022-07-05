#!/usr/bin/env bash

# After installing the correct wheel, use this to install
# the modified ray code.
# See: https://docs.ray.io/en/latest/ray-contribute/development.html

set -e

git clone --depth 1 -b defense/filter-updates-ray-1.13.0 --single-branch https://github.com/HumanCompatibleAI/ray.git ray
cd ray

# This replaces `<package path>/site-packages/ray/<package>`
# with your local `ray/python/ray/<package>`.
python3 python/ray/setup-dev.py -y

cd ..