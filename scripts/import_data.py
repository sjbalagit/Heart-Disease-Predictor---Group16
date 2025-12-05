# download_data.py
# author: Sarisha Das
# date: 2025-12-01

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.read_zip import read_zip

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")
@click.option('--zip-name', type=str, help="Filename for stored zipfile (Uses source filename as default)")

def main(url, write_to, zip_name):
    """Downloads data zip data from the web to a local filepath and extracts it."""

    if not os.path.exists(write_to):
        os.makedirs(write_to)

    try:
        read_zip(url, write_to, zip_name)
    except Exception as e:
        print("There the following error in downloading the zip file", str(e))

if __name__ == '__main__':
    main()