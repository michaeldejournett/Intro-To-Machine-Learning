"""Download the Kaggle dataset into the default kagglehub cache (requires network)."""

from __future__ import annotations

import kagglehub

if __name__ == "__main__":
    path = kagglehub.dataset_download("ravi72munde/uber-lyft-cab-prices")
    print(path)
