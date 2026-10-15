"""Run the experiment pipeline with ``python -m src``."""

import multiprocessing as mp

from .pipeline import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
