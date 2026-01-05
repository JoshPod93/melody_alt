#!/usr/bin/env python
"""
UPIC - Python Clone

A Python implementation of the UPIC (Unité Polyagogique Informatique du CEMAMu)
system, originally created by Iannis Xenakis.

Run this file to start the application.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.main_window import run_app


if __name__ == "__main__":
    run_app()

