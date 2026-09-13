# SAR + Optical Joint Analysis Pipeline

## What this is
Part of a Smart India Hackathon project (SatQuery AI - vision-language assistant
for satellite imagery). This module handles the "optical + SAR joint analysis"
branch: given a co-registered Sentinel-2 (optical) + Sentinel-1 (SAR) image pair
and a natural language query, it fuses the two modalities and answers the query.

A sibling module (change detection) already exists and works the same way -
this one follows the identical pattern so both can plug into the same router.

## Current status
- All 6 pipeline stages are implemented and tested end to end on a synthetic
  test pair (see `data/loader.py -> make_synthetic_pair()`).
- No real Sentinel-1/2 data has been tested yet.
- No real VLM is connected yet - `analysis/vlm_analyzer.py` falls back to a
  basic statistics-based heuristic description when `vlm_call` is None.

## Project structure
```
sar_optical/
├── data/loader.py           # Stage 1: load optical+SAR pair (real files or synthetic)
├── quality/checker.py       # Stage 2: alignment, cloud cover, SAR noise checks
├── preprocessing/prep.py    # Stage 3: despeckle SAR, normalize both images
├── fusion/fuse.py           # Stage 4: combine SAR+optical into one composite image
│                               (3 methods: channel_replace, alpha_blend, edge_overlay)
├── analysis/vlm_analyzer.py # Stage 5: send composite+query to shared VLM (or fallback)
├── output/result_gen.py     # Stage 6: package final answer + confidence note
└── pipeline.py               # Single entry point: run_sar_optical_analysis(...)
```

## The function that matters most
```python
run_sar_optical_analysis(optical_img, sar_img, query, fusion_method="channel_replace", vlm_call=None) -> dict
```
This is what the team's router agent calls. Don't change this function's
signature - only what happens inside it.

## What needs to happen tonight, in order

1. **Confirm it still runs**: `pip install -r requirements.txt --break-system-packages`
   then `python3 pipeline.py`. Should print an Answer and Confidence note using
   the synthetic test pair.

2. **Get real Sentinel-1 + Sentinel-2 image pairs** of the same location (e.g.
   from Copernicus Browser or Google Earth Engine). Swap them into `pipeline.py`'s
   `__main__` block using `load_pair_from_files(optical_path, sar_path)`.

3. **Compare all 3 fusion methods** on the real pair - save each composite
   image to disk and visually compare which preserves the most useful
   information for a VLM to interpret.

4. **Connect a real VLM** as the `vlm_call` parameter. Signature must be:
   `vlm_call(image: np.ndarray, prompt: str) -> str`. Use whatever VLM
   (BLIP-2 / LLaVA / GeoChat) the team is already using for the VQA/captioning
   branch - don't build a separate one if one already exists.

5. **Test with real questions** ("what structures are visible here", "describe
   this area") and check whether the VLM's answers are coherent and actually
   use the SAR-derived information, not just the optical image.

6. **If VLM answers are weak or ignore SAR content**: try a different fusion
   method (edge_overlay distorts colors the least), or rewrite the prompt in
   `analysis/vlm_analyzer.py` to be more explicit. If nothing works in time,
   the heuristic fallback is an acceptable safety net for the live demo -
   it's honest and always produces a sensible-looking answer.

7. **Hand off to the router integration**: confirm whoever is building the
   router agent can call `run_sar_optical_analysis(...)` and get a usable
   result dict back without needing to understand the internals.

## Known limitations to be upfront about
- The quality gate's alignment check only verifies image dimensions match,
  not that the two images are actually the same real-world location - assumes
  pre-registered input, which is a reasonable prototype-stage assumption.
- Fusion is done via a composite image (not true feature-level multimodal
  fusion) - this is clearly noted in the output's `confidence_note` field
  and should be mentioned honestly in the demo/PPT as a scalability item.
