"""Run the publication analyses from any working directory."""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / 'scripts'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figures', action='store_true', help='Rebuild all quantitative figures, including the two supplementary figures.')
parser.add_argument('--reliability', action='store_true', help='Run the full 2,000-bootstrap historical and development-planning analyses.')
args = parser.parse_args()


def run(name, *arguments):
    subprocess.run([sys.executable, str(SCRIPTS / f'{name}.py'), *arguments], check=True)


run('reproduce')
run('verify_results')
if args.figures:
    for name in ['figure02','figures03_05','figure04','figure06','figures07_s1','figures08_09','figureS2']:
        run(name)
if args.reliability:
    run('reliability')
    run('reliability', '--development-planning')
