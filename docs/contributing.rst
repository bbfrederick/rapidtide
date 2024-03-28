Contributing to rapidtide
=========================

This document explains how to set up a development environment for contributing
to rapidtide and code style conventions we follow within the project.
For a more general guide to rapidtide development, please see our
`contributing guide`_. Please also remember to follow our `code of conduct`_.

.. _contributing guide: https://github.com/bbfrederick/rapidtide/blob/main/CONTRIBUTING.md
.. _code of conduct: https://github.com/bbfrederick/rapidtide/blob/main/CODE_OF_CONDUCT.md

Style Guide
-----------

Code
####

Docstrings should follow `numpydoc`_ convention. We encourage extensive
documentation.

The code itself should follow `PEP8`_ convention as much as possible, with at
most about 500 lines of code (not including docstrings) per file*.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/

\* obviously some of the existing files don't conform to this - working on it...

Pull Requests
#############

We encourage the use of standardized tags for categorizing pull requests.
When opening a pull request, please use one of the following prefixes:

    + **[ENH]** for enhancements
    + **[FIX]** for bug fixes
    + **[TST]** for new or updated tests
    + **[DOC]** for new or updated documentation
    + **[STY]** for stylistic changes
    + **[REF]** for refactoring existing code

Pull requests should be submitted early and often!
If your pull request is not yet ready to be merged, please also include the **[WIP]** prefix.
This tells the development team that your pull request is a "work-in-progress",
and that you plan to continue working on it.

A note on current coding quality and style
------------------------------------------

This code has been in active development since June of 2012.  This has two
implications.  The first is that it has been tuned and refined quite a bit over
the years, with a lot of optimizations and bug fixes - most of the core routines
have been tested fairly extensively to get rid of the stupidest bugs.  I find
new bugs all the time, but most of the showstoppers seem to be gone.  The
second result is that the coding style is all over the place.  When I started
writing this, I had just moved over from C, and it was basically a mental port
of how I would write it in C (and I do mean just "C".  Not C++, C#, or anything like
that.  You can literally say my old code has no Class (heh heh)), 
and was extremely unpythonic (I’ve been told by a
somewhat reliable source that looking over some of my early python efforts
“made his eyes bleed”).  Over the years, as I've gone back and added functions,
I periodically get embarrassed and upgrade things to a somewhat more modern
coding style.  I even put in some classes - that's what the cool kids do, right?
But the pace of that effort has to be balanced with the fact that when I make
major architectural changes, I tend to break things.  So be patient with me,
and keep in mind that you get what you pay for, and this cost you nothing!
Function before form.
