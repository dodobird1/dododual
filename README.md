# DoDoDual 🕊️⚡

## NOTE: Currently this model is still under development and doesn't work. Move to https://github.com/dodobird1/dodosu-gen/ for one that does.
**DoDoDual** is an efficient, two-stage deep learning framework for music game chart generation, currently specifically targeting **osu!mania** - for now. Developed by **dodobird1** as an improved continuation of **dodosu-gen**, it leverages a decoupled architecture to first detect rhythmic onsets and then generate complex patterns, making the chart generation process both stable and high-quality.
> ⚠️ **USE THIS MODEL RESPONSIBLY**  
> Disclose any use of AI in the creation of beatmaps. The creator of this model is not responsible for any consequences caused by using this model, especially for plagiarism or any kind of violation of copyright.

## NOTE: The current model's inference script is unusable. I will fix that in a few days. 

## 🌟 Overview

Unlike single-pass models that often struggle with timing precision and pattern variety, DoDoDual splits the task into two specialized components:

1.  **Stage 1: Onset Detector** (Rhythm Phase)
    *   **Architecture**: CNN + Bi-LSTM / Transformer.
    *   **Goal**: Processes raw audio (via Mel Spectrograms) to identify precisely *when* hit objects should occur.
2.  **Stage 2: Pattern Generator** (Placement Phase)
    *   **Architecture**: Transformer-based GAN (Generative Adversarial Network).
    *   **Goal**: Takes the identified timing points and local audio features to decide the column placement (e.g., 4K/7K) and note types (hits/holds).

---

## 🙏 Citing

If you find this code useful in your research or projects, even just a quick "thank you" or a link back to this repository is **greatly appreciated** (but not required under the lisence)! 

**Recommended Citation:**
```bibtex
@misc{dododual2025,
  author = {dodobird1},
  title = {DoDoDual: A Two-Part Efficient Music Game Chart Generation Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/dodobird1/dododual}}
}
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ and the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Training

You can train both stages sequentially or individually using `train_v2.1.py`:

```bash
# Train both onset and pattern models
python train/train.py --stage both --data_dir /path/to/osu/songs/ --save_dir ./model/

# Train only the pattern generator (if onset model is already trained)
python train/train.py --stage pattern --data_dir /path/to/osu/songs/
```

### Inference

Generate a `.osu` chart from any audio file:

```bash
python infer/infer.py --audio "your_song.mp3" \
    --onset_model model/onset_latest.pt \
    --pattern_model model/pattern_gen_latest.pt \
    --keys 4 \
    --osz
```
*Use the `--osz` flag to automatically package the result into a playable osu! archive.*

## 🛠️ Architecture Details

*   **Audio Preprocessing**: Uses `torchaudio` to convert audio into normalized Mel Spectrograms.
*   **MelSpec Module**: A custom module adapted from F5-TTS for high-fidelity audio feature extraction.
*   **GAN Training**: The Pattern Generator uses a Discriminator to ensure generated patterns "look" like human-made charts, improving flow and readability.

## 📈 Potential Improvements

*   **Note Density Control**: Currently, notes can sometimes be generated too close together. Post-processing heuristics or VAE-based approaches could improve spacing.
*   **Data Augmentation**: Implementing pitch-shifting and time-stretching to improve model robustness.
*   **Architectural Upgrades**: Exploring Diffusion Transformers (DiT) or Stable Diffusion-like architectures for the pattern generation stage.
*   **Pattern Variety**: Fine-tuning on specific map styles (e.g., Stream, Jumpstream, Jackhammers).

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Roadmap

- [x] Core model and training pipeline
- [ ] Superparam optimization
- [ ] Data augmentation
- [ ] Non-4K formats (5K, 7K, etc.)
- [ ] GUI application
- [ ] osu!taiko support
- [ ] osu!standard support
- [ ] osu!catch support

---

## Acknowledgements

*No meaning implied by the order of listing.*

- **osu!** — For keeping such a nice, warm, open-source community
- **Salty Mermaid** — From the osu! community, who provided a list of all 2024 ranked and loved beatmaps which served as the training set; they are also working on the 2025 set, which I am also hoping to use. 
- **DiffRhythm & Tencent Music Entertainment (TME) Group** — For introducing me to Music+AI and all its possibilities
- **Mr. Xinning Zhang** — For his excellent AI class!
- **PerseverantDT** — For their JS-based parser of .osu files on GitHub

---

Developed with ❤️ by **dodobird1**.
