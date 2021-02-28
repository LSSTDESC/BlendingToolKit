btk.draw_blends module
=======================
The draw_blends module defines the :class:`~btk.create_blend_generator.DrawBlendsGenerator` class, which is used for generating blended stamps. It also features implementations of this class for WeakLensingDeblending and COSMOS. A custom class may be created by an experienced user to modify the way blends are rendered ; we advise to modify only the :meth:`render_single` method if possible.

.. automodule:: btk.draw_blends
