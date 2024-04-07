# MachLearnEase

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MachLearnEase is a Python package designed to simplify machine learning tasks by providing easy-to-use tools and utilities.

## Installation

You can install MachLearnEase using pip:

```bash
pip install MachLearnEase
```

## Usage

MachLearnEase provides a variety of tools to streamline your machine learning workflow. Here's an example of how to use the `MissingValueImputer` and `OutlierRemoverScaler` classes:

```python
from mlease import MissingValueImputer, OutlierRemoverScaler
import pandas as pd

# Example usage of MissingValueImputer
imputer = MissingValueImputer(strategy='mean')
data = {'A': [1, 2, None, 4], 'B': [5, None, 7, 8]}
df = pd.DataFrame(data)
imputed_df = imputer.fit_transform(df)

# Example usage of OutlierRemoverScaler
scaler = OutlierRemoverScaler()
transformed_data = scaler.fit_transform(your_data_here)
```

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before getting started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- List any contributors or libraries you used or were inspired by.
