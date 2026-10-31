"""Download any missing data and rebuild the modelling table.

Usage:
    python -m src.run_all       # first run ~15 min (mostly API calls), then seconds from the cache

Steps, in order:
1. src.io: prescribing, registered patients, weather and practice locations into dataset/.
2. src.preprocess: join them into dataset/processed/model_table.csv.
Modelling (src.model) and charts (src.graph) are not written yet.
"""

import logging

from . import io, preprocess


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    io.download_all()
    io.save_dataset(preprocess.build_dataset(), io.DATA_DIR / "processed" / "model_table.csv")


if __name__ == "__main__":
    main()
