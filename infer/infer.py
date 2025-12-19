# Modified inference.py — supports single-checkpoint RhythmNet OR two-checkpoint pipeline (onset + pattern)
import torch
import torchaudio
import argparse
import os
import sys
import numpy as np
import zipfile
from tqdm import tqdm

CONFS = [0.7, 0.8, 0.9, 0.97]

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Try imports that may exist in your repo. We'll also define fallback model classes
try:
    from train.train import RhythmNet  # optional: if you have a single-model class
except Exception:
    RhythmNet = None

try:
    from reference.modules import MelSpec
except Exception as e:
    print(f"Error importing MelSpec from reference.modules: {e}")
    raise

# ---------------------------
# Local copies of model classes used in train_v2
# (These must match architecture used in training)
# ---------------------------
import torch.nn as nn

class OnsetDetector(nn.Module):
    """Same architecture as in train_v2.py for onset detection (stage 1)."""
    def __init__(self, n_mels=100, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2) # Reduce time dim by 2
        )
        self.lstm = nn.LSTM(128, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 64), nn.ReLU(),
            nn.Linear(64, 1) # Output 1 probability (Is there a note?)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1).contiguous()  # [B, T', C]
        x, _ = self.lstm(x)
        logits = self.classifier(x)  # [B, T', 1]
        return logits


class PatternGenerator(nn.Module):
    """Same as train_v2 PatternGenerator — expects per-note mel_events (B, Seq, 100)."""
    def __init__(self, input_dim=100, hidden_size=256, keys=4, num_layers=3):
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, hidden_size)
        self.delta_proj = nn.Linear(1, hidden_size)
        self.bpm_proj = nn.Linear(1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, keys)
    def forward(self, mel_events, time_deltas, bpm, mask=None):
        # mel_events: [B, Seq, input_dim], time_deltas: [B, Seq, 1], bpm: [B,1]
        x = self.audio_proj(mel_events) + self.delta_proj(time_deltas) + self.bpm_proj(bpm).unsqueeze(1)
        x = x.contiguous()
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)  # [B, Seq, Keys]

# ---------------------------
# Utilities: load audio -> mel
# ---------------------------
def load_audio(audio_path, sample_rate=24000, hop_length=256, n_mels=100):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    mel_extractor = MelSpec(target_sample_rate=sample_rate, hop_length=hop_length, n_mel_channels=n_mels, normalize=True)
    with torch.no_grad():
        mel = mel_extractor(waveform)  # expected [1, Mels, Time]
        mel = mel.float().contiguous()
    return mel  # shape [1, n_mels, T]

# ---------------------------
# Model loader: robust to different checkpoint types
# ---------------------------
def load_model_flexible(model_path, keys, device):
    """
    Try to load as:
      1) RhythmNet (single-model)
      2) PatternGenerator (stage 2 generator)
      3) OnsetDetector (not used alone but allow loading)
    Returns (model_obj, model_type_str)
    """
    ckpt = torch.load(model_path, map_location=device)
    # If checkpoint is a dict with 'model_state_dict' we try to use its content
    state_dict = None
    if isinstance(ckpt, dict):
        # Accept both raw state_dict and wrapped checkpoint
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            # Maybe training saved raw state_dict inside dict-like structure
            # Heuristic: if keys look like 'audio_proj.weight' it's a PatternGenerator
            possible_keys = set(ckpt.keys())
            # If many keys present and shapes map, treat ckpt as state_dict
            state_dict = ckpt

    # Try RhythmNet first (if available)
    if RhythmNet is not None:
        try:
            model = RhythmNet(keys=keys).to(device)
            model.load_state_dict(state_dict if state_dict is not None else ckpt)
            print("Loaded checkpoint as RhythmNet single-model.")
            return model, 'rhythmnet'
        except Exception as e:
            # fallback
            print("Failed loading as RhythmNet:", e)
            pass

    # Try PatternGenerator (stage 2)
    try:
        model = PatternGenerator(keys=keys).to(device)
        model.load_state_dict(state_dict if state_dict is not None else ckpt)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"CRITICAL ERROR: Model parameter {name} contains NaNs!")
                raise Exception("NaN detected in parameters")
        print("Loaded checkpoint as PatternGenerator (stage 2).")
        return model, 'pattern_generator'
    except Exception as e:
        print("Failed loading as PatternGenerator:", e)
        pass

    # Try OnsetDetector
    try:
        model = OnsetDetector().to(device)
        model.load_state_dict(state_dict if state_dict is not None else ckpt)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"CRITICAL ERROR: Model parameter {name} contains NaNs!")
                raise Exception("NaN detected in parameters")
        print("Loaded checkpoint as OnsetDetector (stage 1).")
        return model, 'onset_detector'
    except Exception as e:
        print("Failed loading as OnsetDetector:", e)
        pass

    raise RuntimeError(f"Unable to auto-detect model type for checkpoint: {model_path}")

# ---------------------------
# Inference functions
# ---------------------------
def generate_from_single_model(model, mel_spec, bpm, device='cpu', threshold=0.5, inference_mode='binary', topk=1, shuffle_prob=0.0, keys=4):
    """
    Backwards-compatible: if model is RhythmNet-like and expects mel_spec + bpm -> per-frame logits.
    """
    model.eval()
    mel_spec = mel_spec.to(device).contiguous()
    with torch.no_grad():
        bpm_tensor = torch.tensor([[bpm]], dtype=torch.float32, device=device)
        logits = model(mel_spec, bpm_tensor, mask=None)  # expect [1, T, Keys]
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    # Apply same conversion modes as earlier: binary/topk/heuristic
    predictions = np.zeros_like(probs, dtype=np.int32)
    if inference_mode == 'binary':
        predictions = (probs > threshold).astype(np.int32)
    elif inference_mode == 'topk':
        k = int(max(1, min(topk, probs.shape[1])))
        for t in range(probs.shape[0]):
            idxs = np.argsort(probs[t])[::-1][:k]
            predictions[t, idxs] = 1
    else:
        # heuristic (legacy)
        for frame_idx in range(probs.shape[0]):
            frame_probs = probs[frame_idx]
            above = frame_probs > threshold
            if above.any():
                mp = frame_probs.max()
                if mp > CONFS[3]:
                    ms = 4
                elif mp > CONFS[2]:
                    ms = 3
                elif mp > CONFS[1]:
                    ms = 2
                else:
                    ms = 1
                sidx = np.argsort(frame_probs)[::-1]
                cnt = 0
                for key in sidx:
                    if frame_probs[key] > threshold and cnt < ms:
                        predictions[frame_idx, key] = 1
                        cnt += 1
    if shuffle_prob > 0:
        predictions = _shuffle_columns(predictions, shuffle_prob, keys)
    return predictions, probs

def detect_onsets_and_generate(onset_model, pattern_model, mel_spec, bpm, device='cpu',
                                onset_threshold=0.5, max_pool_factor=2, keys=4, shuffle_prob=0.0):
    """
    Two-stage pipeline:
      1) onset_model: take mel_spec [1, Mels, T] -> outputs onset logits at pooled rate T' (due to MaxPool1d(2)).
         We threshold (or peak-pick) to get T' indices; map back to mel frame indices by *max_pool_factor.
      2) For each selected mel frame index, extract mel vector (100 dims) -> build mel_events [1, Seq, 100]
         and time_deltas [1, Seq,1], bpm [1,1], feed to pattern_model -> per-note logits [1, Seq, Keys]
      3) Assemble per-frame per-key predictions (place pattern outputs at the original mel frame indices)
    """
    onset_model.eval()
    pattern_model.eval()
    device_t = torch.device(device)
    mel_spec = mel_spec.to(device_t).contiguous()
    with torch.no_grad():
        # Run onset model
        onset_logits = onset_model(mel_spec)  # [1, T', 1]
        onset_probs = torch.sigmoid(onset_logits).cpu().numpy()[0, :, 0]  # [T']
        # Pick indices where prob > threshold
        idxs_pooled = np.where(onset_probs > onset_threshold)[0]
        if idxs_pooled.size == 0:
            # Fall back: take top-k peaks (to avoid empty map)
            topk = min(64, max(1, int(mel_spec.shape[2] // 10)))
            idxs_pooled = np.argsort(onset_probs)[::-1][:topk]
            idxs_pooled = np.sort(idxs_pooled)

        # Map pooled indices back to mel frame indices
        frame_indices = (idxs_pooled * max_pool_factor).astype(int)
        # clamp to valid range
        frame_indices = np.clip(frame_indices, 0, mel_spec.shape[2]-1)

        # Extract mel per-event features
        # mel_spec shape: [1, n_mels, T] -> we take mel_spec[0, :, i] -> vector length n_mels
        mel_np = mel_spec[0].cpu().numpy().T  # [T, n_mels]
        if len(frame_indices) == 0:
            # nothing detected, return empty
            seq_len = 0
            return np.zeros((mel_spec.shape[2], keys), dtype=np.int32), np.zeros((mel_spec.shape[2], keys), dtype=np.float32)

        event_feats = torch.tensor(mel_np[frame_indices], dtype=torch.float32, device=device_t).unsqueeze(0)  # [1, Seq, n_mels]

        # Build time deltas in seconds
        # hop_length/sample_rate are assumed 256/24000 as used in training
        frame_times = frame_indices.astype(np.float32) * (256.0 / 24000.0)
        if len(frame_times) == 1:
            time_deltas = torch.tensor([[0.0]], dtype=torch.float32, device=device_t).unsqueeze(0)  # [1,1,1]
        else:
            diffs = np.concatenate(([0.0], frame_times[1:] - frame_times[:-1])).astype(np.float32)
            time_deltas = torch.tensor(diffs, dtype=torch.float32, device=device_t).unsqueeze(0).unsqueeze(-1)  # [1, Seq, 1]

        bpm_tensor = torch.tensor([[bpm]], dtype=torch.float32, device=device_t)  # [1,1]
        # mask: all valid
        mask = torch.ones((1, event_feats.shape[1]), dtype=torch.bool, device=device_t)

        # Run pattern generator
        logits = pattern_model(event_feats, time_deltas, bpm_tensor, mask=mask)  # [1, Seq, Keys]
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # [Seq, Keys]

        # Convert per-event probs to per-frame predictions (binary threshold 0.5)
        # Place predictions at respective frame indices
        predictions = np.zeros((mel_spec.shape[2], keys), dtype=np.int32)
        probs_full = np.zeros((mel_spec.shape[2], keys), dtype=np.float32)
        for i, fi in enumerate(frame_indices):
            # simple threshold 0.5 here; could be exposed as arg
            preds_i = (probs[i] > 0.5).astype(np.int32)
            predictions[fi] = preds_i
            probs_full[fi] = probs[i]
    # optional shuffle handled by caller
    if shuffle_prob > 0:
        predictions = _shuffle_columns(predictions, shuffle_prob, keys)
    return predictions, probs_full

def _shuffle_columns(predictions, shuffle_prob, num_keys):
    result = predictions.copy()
    for frame_idx in range(result.shape[0]):
        occupied = np.where(result[frame_idx] == 1)[0]
        if len(occupied) == 0 or len(occupied) >= num_keys:
            continue
        all_cols = set(range(num_keys))
        occupied_set = set(occupied)
        for col in list(occupied):
            if np.random.random() < shuffle_prob:
                available = list(all_cols - occupied_set)
                if available:
                    new_col = np.random.choice(available)
                    result[frame_idx, col] = 0
                    result[frame_idx, new_col] = 1
                    occupied_set.remove(col)
                    occupied_set.add(new_col)
    return result

# ---------------------------
# OSU output helper (same as before)
# ---------------------------
def write_osu_file(predictions, audio_path, output_path, bpm, keys=4, hop_length=256, sample_rate=24000,
                   threshold=None, high_conf=None, mid_conf=None):
    filename = os.path.basename(audio_path)
    name_no_ext = os.path.splitext(filename)[0]
    beat_len = 60000 / bpm
    param_str = ""
    if high_conf is not None and mid_conf is not None and threshold is not None:
        param_str = f"Params: high_conf={high_conf}, mid_conf={mid_conf}, thresh={threshold}"
    else:
        param_str = "Params: (not specified)"
    header = f"""osu file format v14

[General]
AudioFilename: {filename}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1.2
BeatDivisor: 4
GridSize: 8
TimelineZoom: 2

[Metadata]
Title:{name_no_ext}
TitleUnicode:{name_no_ext}
Artist:(not set)
ArtistUnicode:(not set)
Creator:dodosu-gen!mania v0.0.1
Version:AI Difficulty (BPM {bpm}, {param_str})
Source:
Tags:AI generated rhythmnet
BeatmapID:0
BeatmapSetID:0

[Difficulty]
HPDrainRate:8
CircleSize:{keys}
OverallDifficulty:8
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Sound Samples

[TimingPoints]
0,{beat_len},4,2,1,60,1,0


[HitObjects]
"""
    hit_objects = []
    frame_time = hop_length / sample_rate
    for frame_idx in range(predictions.shape[0]):
        for key in range(keys):
            if int(predictions[frame_idx, key]) == 1:
                time_ms = int(frame_idx * frame_time * 1000)
                x = int((key * 512) / keys + (512 / keys) / 2)
                y = 192
                line = f"{x},{y},{time_ms},1,0,0:0:0:0:"
                hit_objects.append(line)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n".join(hit_objects))
    print(f"Generated {len(hit_objects)} notes to {output_path}")

def create_osz(osu_path, audio_path, output_path=None):
    if output_path is None:
        output_path = osu_path.rsplit('.', 1)[0] + '.osz'
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(osu_path, arcname=os.path.basename(osu_path))
        zf.write(audio_path, arcname=os.path.basename(audio_path))
    print(f"Created .osz package: {output_path}")
    return output_path

# ---------------------------
# CLI main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate osu!mania beatmap from audio using trained models")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Single-model checkpoint (RhythmNet style).")
    parser.add_argument("--pattern_model", type=str, default=None, help="PatternGenerator checkpoint (stage 2).")
    parser.add_argument("--onset_model", type=str, default=None, help="OnsetDetector checkpoint (stage 1).")
    parser.add_argument("--output", type=str, default="output.osu")
    parser.add_argument("--keys", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold used in 'binary' mode or for pattern generator final threshold.")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--inference_mode", type=str, choices=['binary','topk','heuristic'], default='binary')
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--shuffle", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--osz", action="store_true")
    parser.add_argument("--onset_threshold", type=float, default=0.5, help="Threshold for onset detection (if using onset model).")
    parser.add_argument("--pool_factor", type=int, default=2, help="Pooling factor used by onset model (MaxPool1d) to map pooled indices back to mel frames.")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print("Audio file not found."); sys.exit(1)
    if args.model is None and (args.pattern_model is None or args.onset_model is None):
        print("Either --model (single model) or both --onset_model and --pattern_model must be provided."); sys.exit(1)

    device = torch.device('cpu') if args.cpu else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    # Load audio -> mel
    mel = load_audio(args.audio)  # [1, n_mels, T]
    print(f"Audio frames: {mel.shape[2]}, duration ~ {mel.shape[2]*256/24000:.1f}s")
    mel = mel.to(device).contiguous()

    # Case 1: single model provided
    if args.model:
        if not os.path.exists(args.model):
            print(f"Model checkpoint {args.model} not found."); sys.exit(1)
        model, mtype = load_model_flexible(args.model, keys=args.keys, device=device)
        if mtype != 'rhythmnet':
            print(f"Warning: single --model loaded as '{mtype}', which may not accept full-mel input. Continuing but behavior may differ.")
        preds, probs = generate_from_single_model(model, mel, args.bpm, device=device, threshold=args.threshold, inference_mode=args.inference_mode, topk=args.topk, shuffle_prob=args.shuffle, keys=args.keys)
    else:
        # Two-stage mode
        if not os.path.exists(args.onset_model) or not os.path.exists(args.pattern_model):
            print("onset_model or pattern_model not found."); sys.exit(1)
        onset_model, _ = load_model_flexible(args.onset_model, keys=args.keys, device=device)
        pattern_model, _ = load_model_flexible(args.pattern_model, keys=args.keys, device=device)
        preds, probs = detect_onsets_and_generate(onset_model, pattern_model, mel, args.bpm, device=device, onset_threshold=args.onset_threshold, max_pool_factor=args.pool_factor, keys=args.keys, shuffle_prob=args.shuffle)

    # Write .osu
    write_osu_file(preds, args.audio, args.output, args.bpm, keys=args.keys, threshold=args.threshold, high_conf=None, mid_conf=None)

    if args.osz:
        create_osz(args.output, args.audio)

    total_notes = int(preds.sum())
    duration_sec = mel.shape[2] * 256 / 24000
    nps = total_notes / duration_sec if duration_sec > 0 else 0.0
    print(f"\nStats:\n  Duration: {duration_sec:.1f}s\n  Total notes: {total_notes}\n  Notes per second: {nps:.2f}")
