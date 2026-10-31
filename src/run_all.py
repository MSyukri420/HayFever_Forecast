"""Download any missing data and rebuild the modelling table and features.

Usage:
    python -m src.run_all       # first run ~15 min (mostly API calls), then seconds from the cache

Steps, in order:
1. src.io: prescribing, registered patients, weather and practice locations into dataset/.
2. src.preprocess: join them into dataset/processed/model_table.csv.
3. src.feature: forecasting features (HORIZON months ahead) into dataset/processed/features.csv.
Then run src.model (once per --target) and src.graph separately; see their docstrings.
"""

import logging

from . import feature, io, preprocess


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    io.download_all()
    processed = io.DATA_DIR / "processed"
    table = preprocess.build_dataset()
    io.save_dataset(table, processed / "model_table.csv")
    io.save_dataset(feature.build_features(table), processed / "features.csv")


if __name__ == "__main__":
    main()
