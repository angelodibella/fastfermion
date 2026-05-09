project(
  'fastfermion',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'optimization=3',
    'warning_level=1',
    'b_ndebug=if-release',
  ],
)

py = import('python').find_installation(pure: false)

# --- Dependencies ---

# pybind11: prefer the Python-importable version (pip install pybind11)
# over the system pkg-config one, which may be too old for the
# 3-argument PYBIND11_MODULE (requires >= 2.12).
pybind11_cmd = run_command(
  py, '-c', 'import pybind11; print(pybind11.get_include())',
  check: false,
)
if pybind11_cmd.returncode() == 0
  pybind11_dep = declare_dependency(
    compile_args: ['-isystem', pybind11_cmd.stdout().strip()],
  )
else
  pybind11_dep = dependency('pybind11', required: true)
endif

# OpenMP: optional, enables shared-memory parallelism
omp_dep = dependency('openmp', required: false)

ff_deps = [pybind11_dep]
ff_args = ['-DFF_VERSION="' + meson.project_version() + '"']

if omp_dep.found()
  ff_deps += omp_dep
  ff_args += '-DFF_OPENMP'
  message('OpenMP: enabled')
else
  message('OpenMP: not found, building without shared-memory parallelism')
endif

# --- Extension module ---

ffcore = py.extension_module(
  'ffcore',
  'src/python_bind.cpp',
  subdir: 'fastfermion',
  dependencies: ff_deps,
  include_directories: ['src', 'src/hashmap'],
  cpp_args: ff_args,
  install: true,
)

# --- Python package files ---

py.install_sources(
  'fastfermion/__init__.py',
  'fastfermion/_cirq.py',
  'fastfermion/_of.py',
  'fastfermion/_parse.py',
  subdir: 'fastfermion',
)
