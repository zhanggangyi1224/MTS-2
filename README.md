# MTS (Music-to-Structure) Data Pipeline

End-to-end data prep for long-form, structure-aware text-to-music generation. It loads CCMusic (or a simulated stand-in), augments audio, generates text prompts, organizes splits/configs, builds simulated structure annotations, and writes final reports/CSVs. A batch runner keeps memory in check for large jobs.

## Quick start
```bash
git clone https://github.com/zhanggangyi1224/MTS-2.git
cd MTS-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --config config/config.yaml
```

Batch mode (memory-friendly checkpoints):
```bash
python run_batch_pipeline.py --config config/batch_config.yaml
```

## Key components (src/)
- `pipeline.py` – orchestrates the 6-step flow, outputs summary CSV/MD.
- `data_loader.py` – loads CCMusic or simulated set, extracts features.
- `augmentation.py` – pitch/tempo/noise/EQ/reverb with quality tracking.
- `text_generation.py` – builds prompt pool, assigns/enhances prompts.
- `data_organization.py` – stratified splits and multiple training configs.
- `structure_processing.py` – simulated structure annotations, conditioning data, metrics.
- `batch_processor.py` – checkpointed batch execution with memory monitoring.

## Configuration
- `config/config.yaml` – standard run; `use_simulated_data:true` avoids HF download.
- To use FMA real audio: set `data.use_fma_dataset:true` and point `fma_audio_dir`/`fma_metadata_path` to your local FMA download (e.g., `./fma_data/fma_small`).
- `config/batch_config.yaml` – tuned for batch runs; adjust `batch_processing.batch_size` and `max_memory_percent` to fit your machine.

## Outputs
- `outputs/` (default) contains intermediate JSON/CSV plus `mts_final_dataset.csv`.
- `MTS_Pipeline_Final_Report.md` and `mts_pipeline_complete_summary.json` summarize counts, timings, and file locations.
- Batch runs also emit lightweight checkpoints in `checkpoints/`.

## Notes
- For real CCMusic, set `use_simulated_data:false` and add a Hugging Face token in config.
- Augmented song records no longer store raw audio unless `save_audio` is enabled to keep RAM low.
- If you resume batch runs, checkpoints auto-restore array payloads and stay small by stripping heavy fields. Adjust `heavy_keys` in `batch_processor.py` if you need to persist specific blobs.

## License
See repository license (add if missing).***
