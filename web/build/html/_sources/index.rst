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
    | *the mathematical kernel* | It is for setting up the problem and the               | Natural Python; it works    |
    |                           | discretization theoretically.                          | with Windows, Linux or Mac. |
    +---------------------------+--------------------------------------------------------+-----------------------------+
    | *implementations*         | The discretization is sent to one of the               | Depends on particular       |
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
in the open-access community.


Tutorial
========

The tutorial of *phyem* has four parts,

.. toctree::
   :maxdepth: 1

   tutorial/about
   tutorial/install
   tutorial/documentations
   tutorial/summary


More
====

More examples, demonstrations, explanations, etc., are available.

.. toctree::
   :maxdepth: 1

   jupyter/index
   gallery/index


Contact & contributors
======================

📞 Getting in touch is important.

.. toctree::
   :maxdepth: 1

   contact
   contributors


|

.. error::

    If you find any error in *phyem* or on this site, we appreciate it a lot if you could report it on
    `the Github issue page of phyem <https://github.com/mathischeap/phyem/issues>`_ or let us know
    through :ref:`contact`.

.. admonition:: Privacy statement

   Site traffic is measured only for statistic.
   Absolutely no data which may lead to minimal leak of your privacy is ever recorded.
   Thank you very much for visiting *phyem*.
