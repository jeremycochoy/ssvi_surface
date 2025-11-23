# SSVI Surface

A Python package for modeling implied volatility surfaces using the Stochastic Volatility Inspired (SSVI) model.

## Installation

```bash
pip install -e .
```

## Usage

```python
from ssvi_surface import SSVIModel, forward_bs_price

# Create and fit the model
model = SSVIModel(outside_spread_penalty=2.0, lr=1e-3)
result = model.fit(T_array, strikes_list, bids_list, asks_list, option_types_list, S0, r, q)

# Predict implied volatility
iv = model.predict(k, t)

# Use forward Black-Scholes pricing
price = forward_bs_price(S, K, T, iv, r, q, is_call)
```

## Running Tests

To run the test suite:

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Or run a specific test file
python -m unittest tests.test_ssvi_formula -v
```

The test suite includes property-based tests that verify:
- Mathematical properties of the SSVI formula (positivity, continuity, scaling)
- Theta interpolation accuracy and monotonicity preservation
- Prediction consistency and correctness
- Fitting constraints (arbitrage-free, parameter bounds)
- Integration tests for the full workflow

## Citation

**If you use this software in published research, please cite this repository:**

```bibtex
@software{ssvi_surface,
  title = {SSVI Surface},
  author = {COCHOY, Jeremy},
  year = {2025},
  url = {https://github.com/jeremycochoy/ssvi_surface}
}
```

## References

This implementation is based on the SSVI methodology described in:

```
Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
Quantitative Finance, 14(1), 59-71.
```

## License

MIT License (see LICENSE file)
