.. PHYEM documentation master file, created by
   sphinx-quickstart on Wed Apr 19 11:03:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. image:: https://requires.io/github/giocaizzi/mplStrater/requirements.svg?branch=main
      :alt: Requirements Status
      :target: https://requires.io/github/giocaizzi/mplStrater/requirements/?branch=main


.. _PHYEM:

=======
Welcome
=======

*phyem* is an open-source finite element library that bridges port-based thinking to numerics and enables
**LEGO**-like simulations. The name, *phyem*, stands for

- Python
- port-Hamiltonian
- fem
- physics

It was initiated as a part of the
ERC advanced project `PortWings <http://www.portwings.eu/>`_ awarded to
`prof.dr.ir. Stefano Stramigioli <https://people.utwente.nl/s.stramigioli>`_.

*phyem* is developed under a joint force of
`Andrea Brugnoli <https://www.researchgate.net/profile/Andrea-Brugnoli-3>`_,
`Ramy Rashad <https://ramyrashad.com/>`_,
`Stefano Stramigioli <https://people.utwente.nl/s.stramigioli>`_,
`Yi Zhang <https://mathischeap.com/>`_, and more.

.. The main maintainer is `Yi Zhang <https://mathischeap.com/>`_.


.. _Introduction:

Introduction
============

*phyem* is Python-shelled; users interact with it through Python scripts or in Python console (Python3 only).
The library is made user-friendly such
that only a basic level of Python programming skills is required.


.. important::

    It has two major modules:

    +---------------------------+--------------------------------------------------------+-----------------------------+
    | **module**                | **description**                                        | **Environment**             |
    +---------------------------+--------------------------------------------------------+-----------------------------+
    | *the mathematical kernel* | It is for setting up the problem and the               | natural Python; it works    |
    |                           | discretization theoretically.                          | with Windows, Linux or Mac. |
    +---------------------------+--------------------------------------------------------+-----------------------------+
    | *implementations*         | The discretization is sent to one of the               | depends on particular       |
    |                           | implementations. It uses a particular finite element   | implementations; the        |
    |                           | setting, generates algebraic systems to be solved,     | :ref:`Implementations-msepy`|
    |                           | and eventually gives results to be post-processed.     | implementation              |
    |                           |                                                        | is in natural Python.       |
    +---------------------------+--------------------------------------------------------+-----------------------------+

.. figure:: _static/simple_structure.png
    :width: 100 %

In other words, they represent
the theoretical and numerical aspects of the library, respectively. And the former is pure Python; while the later
has Python shells and APIs to back-ends of different kernels to make fully use of resources
in the open-access community. Read more at

.. toctree::
   :maxdepth: 1

   introduction


.. _Tutorial:

Tutorial
========

.. toctree::
   :maxdepth: 1

   tutorial/install
   tutorial/documentations
   tutorial/summary


.. _Jupyter:

Jupyter notebooks
=================

We sort out jupyter notebooks that demonstrate the using of *phyem* at

.. toctree::
   :maxdepth: 1

   jupyter/index


.. _Gallery:

Gallery
=======

Here, we categorize *predefined objects* (for example, computational domains and meshes) and representative demos,
examples and applications using *phyem* into showcases at

.. toctree::
   :maxdepth: 1

   gallery/index



Contact
=======

Getting in touch is important. See how at

.. toctree::
   :maxdepth: 1

   contact


|

.. ↩️  Back to :ref:`PHYEM`.

.. admonition:: Privacy statement

   Site traffic is measured only for statistic.
   Absolutely no data which may lead to minimal leak of your privacy is ever recorded.
   Thank you very much for visiting *phyem*.
