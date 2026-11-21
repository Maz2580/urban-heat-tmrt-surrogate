"""UMAMI Urban Heat — V3 data pipeline + inference helpers (Phase 1+).

Phase 0/1.6.x V1 inference + V2 inference live in
`app/services/urban_heat_inference.py` (legacy flat layout).
The Phase-1 V3 work (sensor archive, GEE client, Geoscape rasteriser,
patch tensor builder) lives here under a packaged layout so the data
pipeline is testable without dragging the legacy flat module into
every import.
"""
