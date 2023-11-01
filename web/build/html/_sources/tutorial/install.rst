
.. _Install:

Install
=======


.. _Download:

Download
--------

.. important::

    We commend you to visit the release page of *phyem* git repository,
    `Releases <https://github.com/mathischeap/phyem/releases>`_, to download the most recent stable version,
    `v0.0.1 <https://github.com/mathischeap/phyem/releases/tag/v0.0.1-release>`_.


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

The downloaded (and maybe unzipped) library is a folder named ``phyem``. We now call this folder the *package*.

.. hint::

    To make *phyem* library importable, we can do either

    - put the *package* in a dir that is a default system path.
    - put the *package* in a dir which is not a default system path and customize the path locally in Python scripts.

And we recommend the section way to keep the default system path clean.
For example, if the *package* is put in dir :code:`~/my_packages/`, in a Python
script or console, you can do

.. code-block::

    >>> import sys
    >>> ph_dir = '~/my_packages/'
    >>> sys.path.append(ph_dir)

When this script is executed, the path :code:`~/my_packages/` will be added to system path temporally.
Of course, if :code:`~/my_packages/` is a default system path, you can omit above three lines of code.
Then the *phyem* library can be imported by

.. code-block::

    >>> import phyem as ph
    >>> print(ph)

If above commands work, *phyem* is ready in your machine.

.. caution::

    *phyem* is dependent of other Python packages such as numpy, scipy, matplotlib and so on. Check
    :code:`phyem/requirements.text` for the list of dependencies and install whatever you miss through
    for example `pip <https://pypi.org/>`_.

|

↩️  Back to :ref:`PHYEM`.
