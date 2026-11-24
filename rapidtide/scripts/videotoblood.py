#!/usr/bin/env python
"""Standalone entry point for videotoblood."""

import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.videotoblood as theworkflow


def entrypoint():
    pf.generic_init(theworkflow._get_parser, theworkflow.main)


if __name__ == "__main__":
    entrypoint()
