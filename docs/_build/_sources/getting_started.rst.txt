.. _getting_started:

Getting Started
===============


Requirement
-----------

- python and pip (supports python 3.6)
- virtualenv (run ``pip install virtualenv`` or visit `<https://virtualenv.pypa.io/en/stable/installation/>`_)

Installation
------------

Installation from binary
~~~~~~~~~~~~~~~~~~~~~~~~

- The binary file is present at the following location ``bin/``
- Install daggit- 
.. parsed-literal::
    pip install bin/daggit-0.5.0.tar.gz

Installation after build
~~~~~~~~~~~~~~~~~~~~~~~~

- Clone the repository or download the zipped file from `<https://github.com/ekstep/ML-Workbench.git>`_
.. parsed-literal::
    git clone `<https://github.com/ekstep/ML-Workbench.git>`_
- Change directory into ML-Workbench
- Run ``bash build.sh``
- Install - ``pip install bin/daggit-0.5.0.tar.gz``

DAG execution
-------------

Initialize a DAG
~~~~~~~~~~~~~~~~

.. parsed-literal::
    daggit init <path to yaml file>

Run a DAG
~~~~~~~~~

.. parsed-literal::
    daggit run <Experiment_Name>

Seek help
~~~~~~~~~

- Use ``daggit --help`` to know more about the command
- Help on dagit commands can be found using ``daggit <command> --help``


