API Reference
=============

High-level entry points
-----------------------

.. autofunction:: circle_packing_cli.main
.. autofunction:: circle_packing_cli.pack_circles_from_image

Packing algorithms
------------------

.. autofunction:: circle_packing_cli.pack_region_with_circles
.. autofunction:: circle_packing_cli.pack_region_with_circles_dt

Image + color preprocessing
---------------------------

.. autofunction:: circle_packing_cli.center_crop_to_aspect
.. autofunction:: circle_packing_cli.mosaic_labels_by_user_colors
.. autofunction:: circle_packing_cli.map_clusters_to_user_colors

Layout export (CSV/SVG/assembly aids)
-------------------------------------

.. autofunction:: circle_packing_cli.write_layout_csv
.. autofunction:: circle_packing_cli.write_layout_svg
.. autofunction:: circle_packing_cli.write_assembly_aid_csv
.. autofunction:: circle_packing_cli.write_assembly_aid_svg

Utilities
---------

.. autofunction:: circle_packing_cli.announce
.. autofunction:: circle_packing_cli.grid_cell_for
.. autofunction:: circle_packing_cli.round_mm
.. autofunction:: circle_packing_cli.within_bounds
.. autofunction:: circle_packing_cli.circle_fits
.. autofunction:: circle_packing_cli.candidate_points_for_region
.. autofunction:: circle_packing_cli.ensure_bool
