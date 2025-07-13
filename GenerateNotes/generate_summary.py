import os
import subprocess

# === CONFIGURATION ===
LLAMA_MODEL_PATH = '/path/to/your/llama-model.gguf'  # <-- EDIT THIS
LLAMA_BIN_PATH = './bin/main'  # <-- EDIT THIS if needed
WHISPER_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../RenderMermaid/whisper_output.txt')
SUMMARY_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'summary.txt')

# === Read transcript from whisper_output.txt ===
def get_whisper_text():
    if os.path.exists(WHISPER_OUTPUT_PATH):
        with open(WHISPER_OUTPUT_PATH, 'r') as f:
            lines = f.readlines()
        # Remove code block markers if present
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        return ''.join(lines).strip()
    return ''

# === Summarize with Llama LLM ===
def summarize_with_llama(transcript, model_path, llama_bin):
    prompt = (
        "Summarize the following text into a concise list of important points:\n\n"
        f"{transcript}\n\nSummary:"
    )
    try:
        result = subprocess.run(
            [
                llama_bin,
                "-m", model_path,
                "-n", "256",
                "--temp", "0.2",
                "--prompt", prompt
            ],
            capture_output=True,
            text=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error running Llama: {e}")
        return None

# === Main ===
def main():
    transcript = get_whisper_text()
    if not transcript:
        print("No transcript found in whisper_output.txt.")
        return
    if not os.path.exists(LLAMA_MODEL_PATH):
        print(f"Llama model not found: {LLAMA_MODEL_PATH}")
        return
    if not os.path.exists(LLAMA_BIN_PATH):
        print(f"Llama binary not found: {LLAMA_BIN_PATH}")
        return
    print("Generating summary with Llama...")
    summary = summarize_with_llama(transcript, LLAMA_MODEL_PATH, LLAMA_BIN_PATH)
    if summary:
        print("\n=== SUMMARY ===\n")
        print(summary)
        with open(SUMMARY_OUTPUT_PATH, 'w') as f:
            f.write(summary)
        print(f"\nSummary written to {SUMMARY_OUTPUT_PATH}")
    else:
        print("Failed to generate summary.")

if __name__ == "__main__":
    main() 