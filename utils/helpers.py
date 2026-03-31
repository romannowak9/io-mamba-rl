import os


def sort_key_from_filename(filename):
    name = os.path.splitext(filename)[0]
    prefix, number = name.split('_', 1)

    return [prefix, int(number)]