#!/usr/bin/env python
"""Standalone entry point for showrapidtide."""

import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.showrapidtide as theworkflow


def entrypoint():
    pf.generic_init(theworkflow._get_parser, theworkflow.main)


if __name__ == "__main__":
    entrypoint()
