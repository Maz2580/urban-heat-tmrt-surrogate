"""Plan UH-1 — V3 training-data pipeline modules.

  sensor_archive.py — pull historical CoM microclimate readings
  gee_client.py     — server-side GEE service-account wrapper
  patch_builder.py  — assemble per-patch input/label tensors

These run on the local server (no GPU). The trained model lives in
the Colab notebook on Drive; this package only prepares its inputs.
"""
