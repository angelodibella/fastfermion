PYTHON ?= python3
BUILDDIR ?= builddir
MESON := $(PYTHON) -m mesonbuild.mesonmain
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: ffcore test install clean

# Compiles the extension module in place (fastfermion/ffcore...) so that the package can be
# imported from this directory. Pass meson options with MESON_ARGS, e.g. MESON_ARGS=-Dkey_words=1
ffcore:
	$(MESON) setup $(BUILDDIR) $(MESON_ARGS) $$(test -d $(BUILDDIR) && echo --reconfigure)
	$(MESON) compile -C $(BUILDDIR)
	cp $(BUILDDIR)/ffcore$(EXT_SUFFIX) fastfermion/

test: ffcore
	$(PYTHON) -m pytest test

install:
	$(PYTHON) -m pip install .

clean:
	rm -rf $(BUILDDIR) build dist *.egg-info fastfermion/ffcore$(EXT_SUFFIX)

cpptest: src/*.h cpptest/*.h cpptest/run_test.cpp
	g++ -O3 -std=c++2a cpptest/run_test.cpp -o cpptest/run_test
