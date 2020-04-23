"""
Functions for parsers.
"""
import os.path as op


def is_valid_filespec(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is None:
        parser.error('No file specified')

    thesplit = arg.split(':')
    if not op.isfile(thesplit[0]):
        parser.error('The file {0} does not exist!'.format(thesplit[0]))

    return arg


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def invert_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if arg != 'auto':
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

    if arg != 'auto':
        arg = 1.0 / arg
    return arg


def is_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if arg != 'auto':
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

    return arg


def is_int(parser, arg):
    """
    Check if argument is int or auto.
    """
    if arg != 'auto':
        try:
            arg = int(arg)
        except parser.error:
            parser.error('Value {0} is not an int or "auto"'.format(arg))

    return arg


def is_range(parser, arg):
    """
    Check if argument is min/max pair.
    """
    if arg is not None and len(arg) != 2:
        parser.error('Argument must be min/max pair.')
    elif arg is not None and float(arg[0]) > float(arg[1]):
        parser.error('Argument min must be lower than max.')

    return arg
