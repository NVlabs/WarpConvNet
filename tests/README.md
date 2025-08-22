# Testing Guide

## Quick start

Use pytest from the repo root:

```bash
pytest -q
```

GPU tests require CUDA and a built extension (`warpconvnet/_C*.so`).

## Recommended invocations

- Run fast unit tests with progress and summary:

```bash
pytest -vv --maxfail=1
```

- Filter by keyword expression (substring match on test names or node ids):

```bash
pytest -k sparse_conv
pytest tests/csrc -k "wmma and not split_k"
```

- Run a single test file or test function:

```bash
pytest tests/nn/test_sparse_conv.py::test_sparse_conv_amp -vv
```

- Stop on first failure and show local variables on errors:

```bash
pytest -x -vv --showlocals
```

- Re-run only failures from the last run, then all if none failed:

```bash
pytest --lf --maxfail=1
pytest --ff   # run failures first
```

- Use markers to include/exclude groups:

```bash
pytest -m "not slow"
pytest -m csrc
```

Mark a test with `@pytest.mark.csrc` (example) and configure in `pytest.ini` if desired.

- Control CUDA device visibility:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -k sparse_conv
```

## Performance and benchmarking

Some tests use `pytest-benchmark`. To run benchmark groups with more stable timings:

```bash
pytest -k benchmark --benchmark-min-time=0.1 --benchmark-warmup=on
```

Compare two runs:

```bash
pytest -k benchmark --benchmark-save=baseline
pytest -k benchmark --benchmark-compare=baseline
```

## CI-friendly flags

- Disable warnings as errors and reduce output noise:

```bash
pytest -q -ra
```

- Fail on xfails that unexpectedly pass (strict xfail):

```bash
pytest --xfail-strict
```

## Debugging

- Drop into pdb on failure:

```bash
pytest -x -vv --pdb
```

- Use `pytest --maxfail=1` to iterate quickly.

## Tips

- Ensure the Python venv is active and the extension is built:

```bash
source .venv/bin/activate
pytest -q
```

- To rebuild C++/CUDA extension and run impacted tests:

```bash
pip install -e .
pytest tests/csrc -q
```
