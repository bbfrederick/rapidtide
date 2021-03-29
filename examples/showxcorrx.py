# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _ex1:

=========================================
 Run showxcorrx workflow
=========================================

Calculate and display crosscorrelation between two random timeseries.

"""
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Start with the necessary imports
# --------------------------------
from os import remove

import matplotlib.pyplot as plt
import numpy as np

from rapidtide.scripts import showxcorrx

###############################################################################
# Arguments
# --------------------------------
np.random.seed(314)
dat1 = np.random.random(500)
f1 = "f1.txt"
np.savetxt(f1, dat1)

np.random.seed(42)
dat2 = np.random.random(500)
f2 = "f2.txt"
np.savetxt(f2, dat2)

samplerate = 2.0

###############################################################################
# Let's plot the timeseries first
# --------------------------------
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(dat1, color="red", alpha=0.7)
ax.plot(dat2, color="blue", alpha=0.7)
fig.show()

###############################################################################
# Now let's run it
# --------------------------------
showxcorrx.main(f1, f2, samplerate)

###############################################################################
# Clean up
# --------------------------------
remove(f1)
remove(f2)
