"""Kaggle entrypoint for the uploaded script kernel."""
import os
import subprocess
import sys
from pathlib import Path

# Kaggle executes the uploaded kernel payload from its working directory.
repo = Path.cwd()
os.chdir(repo)
requirements = repo / 'requirements-kaggle.txt'
if requirements.exists():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements)])

input_csv = os.environ.get('INPUT_CSV', '/kaggle/input/approved-parallel/parallel.csv')
out = Path('/kaggle/working/santali-output')
out.mkdir(parents=True, exist_ok=True)
subprocess.check_call([sys.executable, 'training/kaggle_prepare_data.py', '--input', input_csv, '--output-dir', str(out / 'data')])
subprocess.check_call([sys.executable, 'training/kaggle_translation_lora.py', '--data-dir', str(out / 'data'), '--output-dir', str(out / 'adapter')])
print('Training artifacts:', out)
