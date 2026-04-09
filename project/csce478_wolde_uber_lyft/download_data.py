"""Downloads the Kaggle zip via kagglehub (needs internet)."""

from __future__ import annotations

import kagglehub

if __name__ == "__main__":
    path = kagglehub.dataset_download("ravi72munde/uber-lyft-cab-prices")
    print(path)
