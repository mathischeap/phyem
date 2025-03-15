
.. _Install:

Installation
============

*phyem* is Python-shelled. You will need a basic level of Python skills to follow this instruction.

.. _Download:

Download
--------

.. important::

    We commend you to visit the release page of *phyem* git repository,
    `Releases <https://github.com/mathischeap/phyem/releases>`_, to download a stable version.


You can also download the most recent (maybe not thoroughly tested) source code from the GitHub page,
`git-phyem <https://github.com/mathischeap/phyem>`_, by, for example, running the following
`git <https://git-scm.com/>`_
command

.. code-block::

    git clone https://github.com/mathischeap/phyem.git

to clone the package to a local repository. Nevertheless, downloading a stable version from the
`Releases <https://github.com/mathischeap/phyem/releases>`_ page is more recommended.


.. _Config:

Config
------

The downloaded library is a folder, and we shall name it ``phyem``.
Note that if you download the
zipped library from github, you got a zipped file, for example, called ``phyem-main``. Then you should un-zip it and
rename the fold to ``phyem``.

We now call this folder the *package*.

.. hint::

    To make *phyem* library importable, we can do either

    - put the *package* in the current working dir.
    - put the *package* in a dir that is a default system path.
    - put the *package* in a dir which is not a default system path and customize the path locally in Python scripts.

For example, if the *package* is put in dir :code:`abs_dir_of_phyem`, in a Python
script or console, you can do

.. code-block::

    >>> import sys
    >>> ph_dir = 'abs_dir_of_phyem'
    >>> sys.path.insert(0, ph_dir)

When this script is executed, the path :code:`abs_dir_of_phyem` will be added to system path temporally.
Of course, if :code:`abs_dir_of_phyem` is a default system path, you can omit above three lines of code.
Then the *phyem* library can be imported by

.. code-block::

    >>> import phyem as ph
    >>> print(ph)

If above commands work, *phyem* is installed in your machine.

.. caution::

    *phyem* is dependent of other Python packages such as numpy, scipy, matplotlib and so on. Check
    :code:`phyem/requirements.text` for the list of dependencies. They are all
    regular Python packages. You can easily install whatever you miss through
    for example `pip <https://pypi.org/>`_.


.. _Test:

Test
----
To test the installation of your *phyem*, you can do

.. code-block::

    >>> ph.test()

This command calls all predefined testing functions to validate functionalities of *phyem* in your machine. If
it raises error, most likely, it is because of a lack in the environment. Just track the exception and install
the missed one(s). When it passes, you are ready to go ahead.

|

↩️  Back to :ref:`PHYEM`.
