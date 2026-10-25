Cross-season Apple-tree Matching
========================================

This project matches Summer RGB/D queries against complete Winter galleries.
RBF-SVR is the main regression method; weighted Linear Regression is the baseline.
The main notebook coordinates the workflow, and reusable implementation lives in
``src/``. A separate development experiment explores optimized features in ``tools/``.

Start with :doc:`quickstart`, use :doc:`workflow` to understand data movement and
evaluation, and consult the :doc:`_generated/index` for exact callable contracts.
The :doc:`data_contract` specifies tables, missing values and experiment artifacts.

.. toctree::
   :maxdepth: 2
   :caption: Project documentation

   quickstart
   workflow
   data_contract
   experiment_reference
   _generated/index
   maintenance
   optimization_plan
   ai_usage

.. note::

   D values are exported 8-bit intensities with an unknown physical scale. Most
   development labels come from filenames. Independent accuracy refers only to
   confirmed queries in the frozen audit, conditional on the fixed Winter gallery.

Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
