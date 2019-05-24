===================
**wavepytools**
===================

`wavepytools <https://github.com/APS-XSD-OPT-Group/wavepytools>`_ contains scripts to run wavepy.

It is not complete, only scripts that are desired to have syncronized in
several machines. Files with data are not tracked.


---------------------
**Syncing with git**
---------------------

.. NOTE:: You need to have ``git`` installed


**Clone**
----------

>>> git clone https://github.com/APS-XSD-OPT-Group/wavepytools.git


**Update your local installation**
----------------------------------

>>> git pull


**To make git to store your credentials**
-----------------------------------------

>>> git config credential.helper store


-----------------------------------
**Solving dependencies with conda**
-----------------------------------

.. NOTE:: You need to have ``anaconda`` installed
https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

After install anaconda, install wavepy

>>~/anaconda3/bin/python3.7 -m pip install wavepy

Install xraylib:

>>>~/anaconda3/bin/conda install -c conda-forge xraylib=3.3.0

Matplotlib version:

The current running version is 3.0.3

**To run scripts in wavepytools**
-----------------------
For example

>>>~/anaconda3/bin/python3.7 ~/wavepytools/wavepytools/imaging/single_grating/singleCheckerboardGratingTalbot_easyqt4.py

