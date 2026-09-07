"""Minimal Gradio chat interface for Pearl-Chat model demonstration."""

from pathlib import Path
from typing import Generator, List, Tuple
from flax import nnx
import gradio as gr

from pearlchat.checkpointing import CheckpointManager
from pearlchat.config import ModelConfig
from pearlchat.generate import generate_stream
from pearlchat.model import PearlChatModel
from pearlchat.tokenizer import LugandaTokenizer


def load_model_and_tokenizer(
    checkpoint_dir: str = "checkpoints/pearlchat-warm-start",
    tokenizer_dir: str = "data/tokenizer",
) -> Tuple[PearlChatModel, LugandaTokenizer]:
    """Load model and tokenizer artifacts."""
    tokenizer_path = Path(tokenizer_dir)
    if not (tokenizer_path / "vocab.json").exists():
        raise FileNotFoundError(
            f"Tokenizer directory {tokenizer_path} does not contain vocab.json."
        )

    tokenizer = LugandaTokenizer.load(tokenizer_path)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        feed_forward_dim=512,
        dropout_rate=0.0,
    )

    rngs = nnx.Rngs(params=42, dropout=42)
    model = PearlChatModel(config=model_config, rngs=rngs)

    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        manager = CheckpointManager(checkpoint_dir=checkpoint_path)
        try:
            model, _ = manager.restore_latest_checkpoint(model)
        except Exception:
            pass

    return model, tokenizer


def build_app(checkpoint_dir: str = "checkpoints/pearlchat-warm-start") -> gr.Blocks:
    """Build minimal Gradio interface."""
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir=checkpoint_dir)

    def respond(
        message: str,
        chat_history: List[Tuple[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
        if not message.strip():
            yield "", chat_history
            return

        chat_history.append((message, ""))
        accumulated_response = ""

        for token in generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=message,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        ):
            accumulated_response += token
            chat_history[-1] = (message, accumulated_response)
            yield "", chat_history

    with gr.Blocks(title="Pearl-Chat") as demo:
        gr.Markdown(
            "# Pearl-Chat\n"
            "Native Luganda language model implemented in pure JAX using Flax NNX."
        )

        chatbot = gr.Chatbot(label="Conversation", height=450)

        with gr.Row():
            msg = gr.Textbox(
                label="Prompt",
                placeholder="Enter prompt in Luganda (for example: Oli otya?)...",
                scale=8,
            )
            submit_btn = gr.Button("Send", scale=1)

        with gr.Accordion("Generation parameters", open=False):
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=16,
                maximum=256,
                value=64,
                step=16,
                label="Maximum new tokens",
            )

        submit_btn.click(
            respond,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[msg, chatbot],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[msg, chatbot],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
