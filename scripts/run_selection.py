"""Local launcher for the published selection experiment."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qnnbench.cli import run

if __name__ == '__main__':
    run('selection')
