"""
Hugging Face Spaces entrypoint for the DRIVE/RLDT Gradio interface.

The script keeps dependencies lightweight and reuses the existing
``create_gradio_interface`` factory so Spaces can discover the app without
command-line flags. Environment variable support mirrors the CLI path in
``RL0910/enhanced_chat_ui.py``.
"""
import os

# Align pandas behaviour with the main UI module before it gets imported.
os.environ.setdefault("PANDAS_USE_PYARROW_BACKEND", "0")
os.environ.setdefault("PANDAS_USE_PYARROW_EXTENSION_ARRAY", "0")

from RL0910.enhanced_chat_ui import create_gradio_interface

# Create the Gradio app for Spaces discovery.
demo = create_gradio_interface()


def main() -> None:
    """Launch the Gradio demo on the port provided by Hugging Face Spaces."""
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
