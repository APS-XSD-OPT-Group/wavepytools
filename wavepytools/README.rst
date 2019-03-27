===================
**pythonWorkspace**
===================

`pythonWorkspace <https://gitlab.com/wcgrizolli/pythonWorkspace>`_ is the
working directory of Walan Grizolli.

It is not complete, only scripts that are desired to have syncronized in
several machines. Files with data are not tracked.


**Link:** https://gitlab.com/wcgrizolli/pythonWorkspace



---------------------
**Syncing with git**
---------------------

.. NOTE:: You need to have ``git`` installed


**Clone**
----------

>>> git clone https://gitlab.com/wcgrizolli/pythonWorkspace.git

.. NOTE:: This is a private project. Your need to have a user at gitlab.com and to be added to the project to have access.


**Update your local installation**
----------------------------------

>>> git pull


**To make git to store your credentials**
-----------------------------------------

>>> git config credential.helper store




-----------------------------------
**Solving dependencies with conda**
-----------------------------------

.. NOTE:: You need to have ``anaconda`` or ``miniconda`` installed


**Creating conda enviroment**
------------------------------

>>> conda create -n ENV_NAME python=3.5 numpy=1.11  scipy=0.17 matplotlib=1.5 spyder=2.3.9 --yes

.. WARNING:: edit ``ENV_NAME``



**Solving dependencies**
------------------------------


Activate the enviroment:

>>> source activate ENV_NAME


.. WARNING:: edit ``ENV_NAME``


>>> conda install scikit-image=0.12 --yes
>>> conda install -c dgursoy dxchange --yes

>>> pip install cffi
>>> pip install unwrap
>>> pip install tqdm
>>> pip install termcolor
>>> pip install easygui_qt

.. NOTE:: ``unwrap`` needs ``cffi``, ``tqdm`` is used for progress bar



**Adding Recomended packages**
------------------------------

>>> conda install -c dgursoy xraylib




**Additional Settings**
-----------------------

``easygui_qt`` conflicts with the Qt backend of
``matplotlib``. The workaround 
is to change the backend to TkAgg. This can be in the *matplotlibrc* file 
(instructions
`here <http://matplotlib.org/users/customizing.html#customizing-matplotlib>`_).
In Spyder this is done in **Tools->Preferences->Console->External Modules**,
where we set **GUI backend** to
**TkAgg**