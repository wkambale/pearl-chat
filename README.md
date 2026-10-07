# Pearl-Chat: Native Luganda language model in pure JAX

Pearl-Chat is a decoder-only causal transformer implemented from scratch in pure JAX using Flax NNX, Optax, Orbax, and Grain. It is trained on native Luganda text with a custom byte-level BPE tokenizer and served via a standalone notebook and a minimal local web application.

The trained model checkpoint and tokenizer are published on the Hugging Face Hub at [kambale/pearl-chat](https://huggingface.co/kambale/pearl-chat).

## Project overview

- Author: Wesley Kambale
- Workshop: "Building Pearl-Chat: Engineering a Native Language Model from Scratch in Pure JAX", PyCon Africa 2026.
- Architecture: Decoder-only Transformer (GPT-2 style) with explicit Flax NNX state.
- Target language: Luganda (`lg`).
- Framework: Pure JAX and Flax NNX. No PyTorch, no TensorFlow.
- Hugging Face repository: `kambale/pearl-chat`

## Model specifications

| Parameter | Value |
| :--- | :--- |
| Model type | Causal autoregressive decoder |
| Parameter count | 8,473,122 trainable parameters (~8.5M parameter model) |
| Vocabulary size | 8,192 byte-level BPE tokens |
| Context length | 256 tokens |
| Layers | 8 transformer decoder blocks |
| Attention heads | 8 heads (head dimension 32) |
| Embedding dimension | 256 |
| Feed-forward dimension | 1024 (4x embedding dimension) |
| Weight tying | Output projection tied to input embedding table |
| Training curriculum | Two-stage curriculum (Stage 1 foundational pre-training + Stage 2 assistant-masked conversational SFT) |
| Training optimizer | Optax AdamW with warmup cosine decay schedule |
| Training throughput | ~100,000 tokens/sec on NVIDIA GPU |

## Training data composition

The model is trained using a two-stage curriculum across four Hugging Face datasets and curated conversational splits:
1. Luganda-English Parallel Corpus (`kambale/luganda-english-parallel-corpus`): Over 25,000 unique contemporary Luganda-English parallel sentence pairs used in Stage 1 foundational pre-training and Stage 2 bilingual translation instruction tuning.
2. Luganda-English Bible Corpus (`kambale/luganda-english-bible-corpus`): Over 30,800 unique Luganda verses (with leading verse numbers and chapter headers stripped) used in Stage 1 foundational pre-training and BPE tokenizer fitting to instill broad Bantu verb morphology and noun-class concords.
3. Luganda Wikipedia (`wikimedia/wikipedia` `20231101.lg` split): Encyclopedic articles covering geography, history, and society in Uganda.
4. Sunbird/salt dataset (`text-all` subset): Strictly filtered English-Luganda parallel translation pairs (`eng_source_text` and `lug_text`).
5. Curated conversational dialogues: Native Luganda greetings, geography queries, courtesies, and question-answer pairs formatted using structured speaker-turn tags (`Omuntu: <query>\nOmuyambi: <response>`) with assistant-only loss masking in Stage 2.

## Repository layout

```
pearl-chat/
  README.md                              # project documentation and guides
  LICENSE                                # Apache 2.0 license
  .gitignore                             # git ignore patterns
  app.py                                 # entry point to launch the local web application
  ui/
    chat_app.py                          # self-contained local inference server and model runtime
    index.html                           # web application front end
    app.js                               # interactive client logic and streaming handler
    style.css                            # custom design system and layout styling
```

## Running the standalone notebook in Google Colab

The repository includes a completely standalone, self-contained notebook. It defines all models, tokenizers, data loaders, training loops, and generation helpers inline. It can be run on Google Colab without cloning the repository or installing custom packages.

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File** -> **Upload the notebook**
3. Set the runtime to **GPU** (T4 or A100) or **TPU**.
4. (Optional) Provide your Hugging Face token in Colab Secrets under `HF_TOKEN` if you wish to push new checkpoints.
5. Select **Runtime** -> **Run all**.

The notebook executes the complete end-to-end lifecycle:
- Accelerator verification in JAX.
- Luganda text acquisition, cleaning, and dialogue oversampling across four datasets (`kambale/luganda-english-parallel-corpus`, `kambale/luganda-english-bible-corpus`, Luganda Wikipedia, and `Sunbird/salt`).
- From-scratch Byte-Level BPE tokenizer training (8,192 vocabulary).
- Deterministic data loading and batch packing with Google Grain.
- GPT-2 model definition (~8.5M parameters, 8 layers, 8 heads) and functional state inspection (`nnx.split` and `nnx.merge`).
- Two-stage training loop (Stage 1 foundational pre-training + Stage 2 assistant-masked conversational fine-tuning) with Optax AdamW and smoothed Exponential Moving Average (EMA) loss tracking.
- Sharding and device mesh demonstration across logical devices.
- Autoregressive text generation using speaker-turn prompting (`Omuntu: ... Omuyambi: ...`).
- Staging and deployment to the Hugging Face Hub, plus offline `.tar.gz` bundle export.
- In-notebook interactive Gradio chat interface.
- Checkpoint restoration and continuous fine-tuning from `kambale/pearl-chat`.

## Local web application

The repository includes a self-contained local web application (`app.py` and `ui/`) for chatting with the model using real-time token streaming.

### Launching the web application

```bash
# Launch server with local checkpoint or Hugging Face Hub model (kambale/pearl-chat)
python3 app.py

# Or specify a custom port or checkpoint
python3 app.py --port 7860 --checkpoint checkpoints/pearl-chat
```

Open your browser and navigate to:
```
http://127.0.0.1:7860/
```

### Features of the web application

- Automatic checkpoint loading: Seamlessly loads from the bundled `checkpoints/pearl-chat` directory or downloads from the Hugging Face Hub (`kambale/pearl-chat`).
- Server-Sent Events (SSE): Streams generated tokens one by one as they are computed.
- Conversational template handling: Formats incoming prompts with `Omuntu: <prompt>\nOmuyambi:` and stops on subsequent turn boundaries.
- Generation parameter controls: Sliders for temperature, top-p (nucleus), top-k, and maximum generated tokens.
- Live performance metrics: Displays token count and generation throughput in tokens per second.
- Suggested Luganda prompts: Instant one-click starters for greetings, geography, translation, and everyday questions.

## Citation

If you use Pearl-Chat or the associated training pipeline in your research or educational materials, please cite:

```bibtex
@misc{kambale2026pearlchat,
  author = {Wesley Kambale},
  title = {Pearl-Chat: Engineering a Native Language Model from Scratch in Pure JAX},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {https://huggingface.co/kambale/pearl-chat},
  note = {PyCon Africa 2026}
}
```

## License

MIT License
