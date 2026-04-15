# Contributing

Contributions are welcome! Please open issues and pull requests on
[GitHub](https://github.com/nvlabs/warpconvnet).

## Development setup

```bash
git clone https://github.com/nvlabs/warpconvnet.git
cd warpconvnet
pip install -e . --no-build-isolation
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

- Format with `black` and `isort`.
- Use type annotations for public APIs.
- Follow [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

## Pull requests

1. Fork the repository and create a feature branch.
2. Add tests for new functionality.
3. Ensure all tests pass before submitting.
4. Open a pull request against `main`.
