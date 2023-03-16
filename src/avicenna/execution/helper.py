import os
import glob
from pathlib import Path


colors = {
    "blue": "\033[94m",
    "green": "\033[92m",
    "red": "\033[91m",
    "reset": "\033[0m",
}


def error(message):
    print(colors["red"] + "[ERROR]" + colors["reset"] + message)


def info(message):
    print(colors["blue"] + "[INFO] " + colors["reset"] + message)


def get_all_files(file_location: Path):
    files = "*.*"  # + extension

    iter_files = file_location / files
    item_list = glob.glob(str(iter_files))
    return item_list


def removesSetFromList(seedlist, items):
    for i in items:
        seedlist.remove(i)
    temp = seedlist
    return temp
