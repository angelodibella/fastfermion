project(
  'fastfermion',
  'cpp',
  version: '0.1.0',
  meson_version: '>=1.2',
  default_options: ['cpp_std=c++20', 'buildtype=release'],
)

py = import('python').find_installation(pure: false)

# pybind11 headers come from the pip package (pybind11 >= 2.13 for mod_gil_not_used)
pybind11 = run_command(
  py, '-c', 'import pybind11; print(pybind11.get_include()); print(pybind11.__version__)',
  check: false,
)
if pybind11.returncode() != 0
  error('pybind11 not found: pip install pybind11')
endif
pybind11_include = pybind11.stdout().split('\n')[0].strip()
pybind11_version = pybind11.stdout().split('\n')[1].strip()
if not pybind11_version.version_compare('>=2.13')
  error('pybind11 >= 2.13 is required, found ' + pybind11_version)
endif

ff_deps = []
ff_args = [
  '-DFF_VERSION="' + meson.project_version() + '"',
  '-DSYS_NUM_ULONG=' + get_option('key_words').to_string(),
]

# OpenMP is optional: without it propagate() runs single-threaded
omp = dependency('openmp', required: false)
if omp.found()
  ff_deps += omp
  ff_args += '-DFF_OPENMP'
endif
message('OpenMP: ' + (omp.found() ? 'enabled' : 'not found, propagate() will be single-threaded'))

# Hide the extension's own symbols (pybind11's recommendation) and, on Linux, those of any
# statically linked runtime: a libstdc++ bundled into the module would otherwise share its
# STB_GNU_UNIQUE symbols with a different libstdc++ loaded by another extension.
link_args = []
if host_machine.system() == 'linux'
  link_args += '-Wl,--exclude-libs,ALL'
endif

py.extension_module(
  'ffcore',
  'src/python_bind.cpp',
  subdir: 'fastfermion',
  dependencies: ff_deps,
  include_directories: [include_directories('src', 'src/hashmap'), include_directories(pybind11_include, is_system: true)],
  cpp_args: ff_args,
  gnu_symbol_visibility: 'hidden',
  link_args: link_args,
  install: true,
)

py.install_sources(
  'fastfermion/__init__.py',
  'fastfermion/_cirq.py',
  'fastfermion/_of.py',
  'fastfermion/_parse.py',
  subdir: 'fastfermion',
)
