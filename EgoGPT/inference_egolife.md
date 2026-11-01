# inference_egolife.py

Batch captioning for the EgoLife dataset with EgoGPT. The script walks every day folder for one participant, captions each 30-second clip (video + audio), and saves timestamped captions to a JSON file that EgoRAG can use.

## Requirements

- EgoGPT installed (`pip install -e .` inside `EgoGPT/`) and a CUDA GPU
- `whisper`, `decord`, `soundfile` (libsndfile ≥ 1.1 for `.mp3`), `scipy`
- The `EgoRAG` folder next to `EgoGPT` (for `prompt_for_caption` and `experiment_logger`)
- EgoGPT-7B checkpoint downloaded locally

## Expected data layout

```
data/EgoLife/A4_LUCIA/
    DAY1/DAY1_A4_LUCIA_11092500.mp4
    DAY2/...
EgoIT_99K/EgoLife_audio/EgoLife_audio/A4_LUCIA/
    DAY1/DAY1_A4_LUCIA_11092500.mp3
    DAY2/...
```

Every video needs an audio file with the same name (`.mp3`). Clips without audio are skipped.

## Usage

Run from the `EgoGPT/` folder:

```bash
python inference_egolife.py \
    --pretrained_path /path/to/models/7b \
    --video_base data/EgoLife/A4_LUCIA \
    --audio_base EgoIT_99K/EgoLife_audio/EgoLife_audio/A4_LUCIA \
    --output Lucia_meta.json \
    --experiment_name meta_7days_LUCIA
```

| Argument | Default | Description |
|---|---|---|
| `--pretrained_path` | local `models/7b` path | EgoGPT checkpoint |
| `--video_base` | `data/EgoLife/A4_LUCIA` | Folder with one subfolder per day |
| `--audio_base` | `EgoIT_99K/.../A4_LUCIA` | Matching audio folder |
| `--query` | `query_for_meta_final` | Captioning prompt (from `EgoRAG/prompt_for_caption.py`) |
| `--output` | `Lucia_meta.json` | Output file; use a new name per run to avoid overwriting |
| `--experiment_name` | `Withoutmeta_7days_TASHA` | Log name; set it to match your run |

## Output

A JSON list, written after every clip so a crash doesn't lose finished work:

```json
[
  {
    "start_time": "11092500",
    "end_time": "11095500",
    "text": "caption from EgoGPT ...",
    "date": "DAY1",
    "video_path": "data/EgoLife/A4_LUCIA/DAY1/DAY1_A4_LUCIA_11092500.mp4"
  }
]
```

Times use the `HHMMSSFF` format from the filename; `end_time` is `start_time + 30 s`.

## Notes

- The model is loaded once and reused for all clips.
- Decoding is greedy (`do_sample=False`), so captions are reproducible.
- Clip length is assumed to be 30 s; shorter final clips get a slightly wrong `end_time`.
- To caption another participant, change `--video_base`, `--audio_base`, `--output` and `--experiment_name`.