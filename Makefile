# fastfermion build via meson
#
# Quick start:
#   pip install meson meson-python pybind11   # build dependencies
#   make build                                # compile C++ extension
#   pip install --no-build-isolation -e .     # install into current env
#   make test                                 # run tests (needs pytest, scipy)
#
# Override the Python interpreter:
#   make PYTHON=/path/to/python build

PYTHON ?= python3
BUILDDIR := builddir

.PHONY: setup build install test clean

setup:
	$(PYTHON) -m mesonbuild.mesonmain setup $(BUILDDIR)

build: setup
	$(PYTHON) -m mesonbuild.mesonmain compile -C $(BUILDDIR)

install:
	pip install --no-build-isolation -e .

test:
	$(PYTHON) -m pytest test/ -v

clean:
	rm -rf $(BUILDDIR) build/ dist/ *.egg-info
	rm -f fastfermion/ffcore*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
