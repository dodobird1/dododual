import os
import sys
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json

# Add the project root to sys.path to allow importing from reference/
# Assuming this file is located in /mnt/code/elec/osu/pretrain/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from utils.osu_mania_parser import parse_beatmap, Beatmap
    from reference.modules import MelSpec
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure 'utils/osu_mania_parser.py' and 'reference/modules.py' exist.")
    sys.exit(1)

class OsuManiaDataset(Dataset):
    def __init__(self, 
                 root_dir: str | list[str], 
                 keys: int = 4, 
                 sample_rate: int = 24000, 
                 hop_length: int = 256,
                 max_duration: int = None,
                 cache_processed: bool = False,
                 stage: str = 'onset'): # 'onset' or 'pattern'
        """
        Args:
            stage: 'onset' returns dense frames for timing detection.
                   'pattern' returns sparse event sequences for finding columns.
        """
        self.root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
        self.keys = keys
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.cache_processed = cache_processed
        self.stage = stage  # Store the stage
        
        self.mel_extractor = MelSpec(
            target_sample_rate=sample_rate,
            hop_length=hop_length,
            n_mel_channels=100,
            normalize=True
        )
        
        # Scan maps
        self.map_entries = self._scan_maps()
        print(f"Dataset initialized for stage: {self.stage}. Found {len(self.map_entries)} maps.")

    def _scan_maps(self):
        entries = []
        for root_dir in self.root_dirs:
            if not os.path.exists(root_dir): continue
            for root, dirs, files in os.walk(root_dir):
                audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
                if not audio_files: continue
                try: main_audio = max(audio_files, key=lambda f: os.path.getsize(os.path.join(root, f)))
                except OSError: continue
                for file in files:
                    if file.endswith(".osu"):
                        try:
                            entry = self._validate_map(os.path.join(root, file), main_audio)
                            if entry: entries.append(entry)
                        except: continue
        return entries

    def _validate_map(self, osu_path, audio_filename):
        audio_path = os.path.join(os.path.dirname(osu_path), audio_filename)
        if not os.path.exists(audio_path): return None
        try: beatmap = parse_beatmap(osu_path)
        except: return None
        if beatmap.key_count != self.keys: return None
        return {"osu_path": osu_path, "audio_path": audio_path, 
                "beatmap_obj": beatmap, 
                "bpm": (beatmap.min_bpm + beatmap.max_bpm)/2.0}

    def __len__(self):
        return len(self.map_entries)

    def __getitem__(self, idx):
        while True:
            entry = self.map_entries[idx]
            
            mel_spec = self._process_audio(entry["audio_path"]) # [100, T]
            num_frames = mel_spec.shape[1]

            dense_labels = self._process_chart(entry["beatmap_obj"], num_frames)
            
            if self.stage == 'onset':
                onset_label = (dense_labels.sum(dim=1) > 0).float().unsqueeze(1)
                return {
                    "mel": mel_spec,
                    "labels": onset_label,
                    "bpm": torch.tensor([entry["bpm"]], dtype=torch.float32),
                    "path": entry["osu_path"]
                }

            elif self.stage == 'pattern':
                note_indices = torch.where(dense_labels.sum(dim=1) > 0)[0]
                
                if len(note_indices) == 0:
                    idx = (idx + 1) % len(self)
                    continue
                
                if len(note_indices) > 512:
                    start_idx = torch.randint(0, len(note_indices) - 512, (1,)).item()
                    note_indices = note_indices[start_idx : start_idx + 512]
                
                mel_events = mel_spec.transpose(0, 1)[note_indices]
                target_events = dense_labels[note_indices]
                
                frame_times = note_indices.float() * (self.hop_length / self.sample_rate)
                time_deltas = torch.cat([torch.tensor([0.0]), 
                                         frame_times[1:] - frame_times[:-1]]).unsqueeze(1)
                
                return {
                    "mel_events": mel_events,
                    "time_deltas": time_deltas,
                    "labels": target_events,
                    "bpm": torch.tensor([entry["bpm"]], dtype=torch.float32)
                }

    # Helper methods
    def _process_audio(self, path):
        try: waveform, sr = torchaudio.load(path)
        except: return torch.zeros((100, 1000))
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate: 
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        with torch.no_grad(): 
            mel = self.mel_extractor(waveform).squeeze(0)
        return mel

    def _process_chart(self, beatmap, num_frames):
        labels = torch.zeros((num_frames, self.keys), dtype=torch.float32)
        frame_time = self.hop_length / self.sample_rate
        for obj in beatmap.hit_objects:
            idx = int((obj.time / 1000.0) / frame_time)
            if idx < num_frames:
                col = int(obj.x * self.keys / 512)
                col = min(max(col, 0), self.keys - 1)
                labels[idx, col] = 1.0
        return labels


def collate_stage2(batch):
    mel_events = pad_sequence([b['mel_events'] for b in batch], batch_first=True)
    time_deltas = pad_sequence([b['time_deltas'] for b in batch], batch_first=True)
    labels = pad_sequence([b['labels'] for b in batch], batch_first=True)
    bpms = torch.stack([b['bpm'] for b in batch])
    
    lengths = torch.tensor([b['mel_events'].shape[0] for b in batch])
    mask = torch.arange(mel_events.shape[1]).expand(len(batch), -1) < lengths.unsqueeze(1)
    
    return {
        "mel_events": mel_events,
        "time_deltas": time_deltas,
        "labels": labels,
        "bpm": bpms,
        "mask": mask
    }


def collate_fn(batch, max_frames=None):
    """
    If max_frames == 0 → treat as unlimited (no cropping).
    """
    # -------------------------------
    # ✔ NEW BEHAVIOR
    # -------------------------------
    if max_frames == 0:
        max_frames = None  # Disable cropping

    batch.sort(key=lambda x: x['mel'].shape[1], reverse=True)
    
    mels = [x['mel'] for x in batch]
    labels = [x['labels'] for x in batch]
    bpms = [x['bpm'] for x in batch]
    paths = [x['path'] for x in batch]
    
    # Apply random cropping *only if max_frames > 0*
    if max_frames is not None and max_frames > 0:
        cropped_mels = []
        cropped_labels = []
        for mel, label in zip(mels, labels):
            seq_len = mel.shape[1]
            if seq_len > max_frames:
                start = torch.randint(0, seq_len - max_frames + 1, (1,)).item()
                mel = mel[:, start:start+max_frames]
                label = label[start:start+max_frames]
            cropped_mels.append(mel)
            cropped_labels.append(label)
        mels = cropped_mels
        labels = cropped_labels
    
    max_len = max(m.shape[1] for m in mels)
    
    padded_mels = torch.zeros(len(batch), mels[0].shape[0], max_len)
    padded_labels = torch.zeros(len(batch), max_len, labels[0].shape[1])
    
    for i,(mel,label) in enumerate(zip(mels, labels)):
        l = mel.shape[1]
        padded_mels[i,:,:l] = mel
        padded_labels[i,:l,:] = label
    
    padded_bpms = torch.stack(bpms)
    lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    
    return {
        "mel": padded_mels,
        "labels": padded_labels,
        "bpm": padded_bpms,
        "lengths": lengths,
        "paths": paths
    }


def make_collate_fn(max_frames=None):
    """
    If max_frames == 0 → disable cropping.
    """
    if max_frames == 0:
        max_frames = None

    def _collate(batch):
        return collate_fn(batch, max_frames=max_frames)

    return _collate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--keys", type=int, default=4)
    args = parser.parse_args()
    
    dataset = OsuManiaDataset(root_dir=args.data_dir, keys=args.keys, cache_processed=False)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample processed:")
        print(f"Mel shape: {sample['mel'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print("Batch shape (Mel):", batch['mel'].shape)
        print("Batch shape (Labels):", batch['labels'].shape)
    else:
        print("No valid maps found.")
 