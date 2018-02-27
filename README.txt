STXM live analysis tool
-----------------------
Provides real-time feedback on STXM data, mainly by
auto-aligning image stacks and processing them into
NEXAFS spectra. Writes files for later analysis
with aXis2000.

Requirements:
Python3
tkinter
numpy
scipy
matplotlib
watchdog
keras with any backend (tensorflow recommended; GPU acceleration not required)

Edit stxmlive_config.txt to set the default top-level directory for data.
