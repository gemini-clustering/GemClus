:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members: set_params, get_params

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div style='clear:both'></div>
