#!/usr/bin/env python3
"""Plot comparison recordings; accepts the options of plot_random_samples.py."""
import sys
from plot_random_samples import OUT_DIR, main

if __name__ == '__main__':
    main(['--comparison-dir', str(OUT_DIR / 'controller_comparison'), *sys.argv[1:]])
