# fastfermion build via meson
#
# Quick start:
#   pip install meson meson-python pybind11   # build dependencies
#   make build                                # compile C++ extension
#   make test                                 # run tests (needs pytest, scipy)
#
# Override the Python interpreter:
#   make PYTHON=/path/to/python build

PYTHON ?= python3
BUILDDIR := builddir
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: setup build install test clean

setup:
	$(PYTHON) -m mesonbuild.mesonmain setup $(BUILDDIR)

build: setup
	$(PYTHON) -m mesonbuild.mesonmain compile -C $(BUILDDIR)
	cp $(BUILDDIR)/ffcore$(EXT_SUFFIX) fastfermion/
	ln -sf $(BUILDDIR)/compile_commands.json compile_commands.json

install:
	pip install --no-build-isolation -e .

test: build
	cd test && $(PYTHON) -m pytest . -v

clean:
	rm -rf $(BUILDDIR) build/ dist/ *.egg-info
	rm -f fastfermion/ffcore*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
