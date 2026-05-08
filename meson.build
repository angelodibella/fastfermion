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

# pybind11: try pkg-config/cmake first, fall back to -I flag via python import
pybind11_dep = dependency('pybind11', required: false)
if not pybind11_dep.found()
  pybind11_cmd = run_command(
    py, '-c', 'import pybind11; print(pybind11.get_include())',
    check: false,
  )
  if pybind11_cmd.returncode() == 0
    pybind11_dep = declare_dependency(
      compile_args: ['-isystem', pybind11_cmd.stdout().strip()],
    )
  else
    error('pybind11 not found. Install it: pip install pybind11')
  endif
endif

ff_version = meson.project_version()

ffcore = py.extension_module(
  'ffcore',
  'src/python_bind.cpp',
  subdir: 'fastfermion',
  dependencies: pybind11_dep,
  include_directories: ['src', 'src/hashmap'],
  cpp_args: ['-DFF_VERSION="' + ff_version + '"'],
  install: true,
)

py.install_sources(
  'fastfermion/__init__.py',
  'fastfermion/_cirq.py',
  'fastfermion/_of.py',
  'fastfermion/_parse.py',
  subdir: 'fastfermion',
)
