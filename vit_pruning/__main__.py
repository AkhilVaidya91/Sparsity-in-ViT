"""Run the experiment pipeline with ``python -m vit_pruning``."""

import multiprocessing as mp

from .pipeline import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
