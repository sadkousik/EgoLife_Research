import argparse
import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import whisper
from decord import VideoReader, cpu
from egogpt.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX
from egogpt.conversation import conv_templates
from egogpt.model.builder import load_pretrained_model
from scipy.signal import resample

# Resolve project root so EgoRAG can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from EgoRAG.FIO.experiment_logger import start_experiment
from prompt_for_caption import query_for_meta_final

# ---------------------------------------------------------------------------
# Paths for EgoLife
# ---------------------------------------------------------------------------
VIDEO_BASE = "/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/datasets/EgoLife/A5_KATRINA"
AUDIO_BASE = "/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/datasets/EgoLife_audio/A5_KATRINA"
DEFAULT_MODEL_PATH = "/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/model/7b"

all_results = []  # stores all caption results


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(rank, world_size):
    """Set up the (single-process) distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356" # chnaged from 12355 to 12356 to avoid port conflict
    if dist.is_initialized():
        return
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def load_model_once(pretrained_path=DEFAULT_MODEL_PATH):
    warnings.filterwarnings("ignore")
    setup(0, 1)
    device = "cuda"
    device_map = "cuda"

    print("GPU memory BEFORE loading model:",
          torch.cuda.memory_allocated() / 1024**3, "GB")

    tokenizer, model, max_length = load_pretrained_model(
        pretrained_path, device_map=device_map
    )
    model.eval()

    print("Allocated GPU memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("Reserved GPU memory:", torch.cuda.memory_reserved() / 1024**3, "GB")
    return tokenizer, model, max_length, device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_video(video_path=None, audio_path=None, max_frames_num=16, fps=1):
    if audio_path is not None:
        speech, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            target_length = int(len(speech) * 16000 / sample_rate)
            speech = resample(speech, target_length)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
    else:
        speech = torch.zeros(3000, 128)
        speech_lengths = torch.LongTensor([3000])

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    if max_frames_num > 0 and len(frame_idx) > max_frames_num:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    return video, speech, speech_lengths


def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    return [part for part in parts if part]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(tokenizer, model, device, video_path, audio_path, query):
    warnings.filterwarnings("ignore")

    conv_template = "qwen_1_5"
    question = f"<image>\n<speech>\n\n{query}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # Load video + audio
    video, speech, speech_lengths = load_video(
        video_path=video_path, audio_path=audio_path
    )
    print(f"Video shape: {video.shape}, Speech shape: {speech.shape}, "
          f"Speech lengths: {speech_lengths}")

    speech = torch.stack([speech]).to(device).half()

    processor = model.get_vision_tower().image_processor
    processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values"]
    image = [(processed_video, video[0].size, "video")]

    # Token construction
    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if part == "<image>":
            input_ids.append(IMAGE_TOKEN_INDEX)
        elif part == "<speech>":
            input_ids.append(SPEECH_TOKEN_INDEX)
        else:
            input_ids.extend(tokenizer(part).input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    image_tensor = [image[0][0].half()]
    image_sizes = [image[0][1]]
    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            speech=speech,
            speech_lengths=speech_lengths,
            do_sample=False,  # deterministic captions for reproducible RAG runs
            max_new_tokens=4096,
            modalities=["video"],
            **generate_kwargs,
        )

    return tokenizer.batch_decode(cont, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# EgoLife metadata
# ---------------------------------------------------------------------------
def parse_video_metadata(video_path, clip_seconds=30):
    """
    Filename example: DAY1_A2_ALICE_11092500.mp4
    The last field is HHMMSSFF. End time = start time + clip_seconds,
    with correct minute/hour rollover.
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    parts = name.split("_")

    date = parts[0]
    start_time = parts[-1]

    h, m, s, f = (int(start_time[i:i + 2]) for i in (0, 2, 4, 6))
    total = h * 3600 + m * 60 + s + clip_seconds
    end_time = (
        f"{total // 3600:02d}{(total % 3600) // 60:02d}{total % 60:02d}{f:02d}"
    )
    return date, start_time, end_time


def get_subdirs(path):
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


def save_results(output_path):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)


# ---------------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------------
def process_folder(tokenizer, model, device, video_dir, audio_dir, query, output_path):
    video_files = sorted(os.listdir(video_dir))

    for video_file in video_files:
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, os.path.splitext(video_file)[0] + ".mp3")

        if not os.path.exists(audio_path):
            print(f"Skipping {video_file}: audio missing")
            continue

        print(f"\nProcessing {video_file}")

        outputs = run_inference(
            tokenizer, model, device, video_path, audio_path, query
        )

        date, start_time, end_time = parse_video_metadata(video_path)
        print(f"Date: {date}, Start Time: {start_time}, End Time: {end_time}")
        print(outputs)

        all_results.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": outputs[0],
            "date": date,
            "video_path": video_path,
        })

        # Save after every clip so a crash doesn't lose earlier captions
        save_results(output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--query", type=str, default=query_for_meta_final)
    parser.add_argument("--video_base", type=str, default=VIDEO_BASE)
    parser.add_argument("--audio_base", type=str, default=AUDIO_BASE)
    parser.add_argument("--output", type=str, default="/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/generated_caption/Katrina_meta.json",
                        help="Change the name for each run to avoid overwriting")
    parser.add_argument("--experiment_name", type=str, default="meta_katrina")
    args = parser.parse_args()

    logger = start_experiment(args.experiment_name)

    # Load model ONCE
    tokenizer, model, max_length, device = load_model_once(args.pretrained_path)

    try:
        for subdir in get_subdirs(args.video_base):
            video_subdir = os.path.join(args.video_base, subdir)
            audio_subdir = os.path.join(args.audio_base, subdir)
            process_folder(
                tokenizer, model, device,
                video_subdir, audio_subdir,
                args.query, args.output,
            )
    finally:
        save_results(args.output)
        print(f"Saved {len(all_results)} captions to {args.output}")