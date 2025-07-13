import time
import os

MERMAID_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Real-Time Mermaid Diagram</title>
    <style>
      body {{ background: #222; color: #eee; font-family: sans-serif; }}
      pre.mermaid {{ background: #222; color: #eee; }}
    </style>
  </head>
  <body>
    <h2>Real-Time Mermaid Diagram</h2>
    <pre class="mermaid">
{mermaid_code}
    </pre>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true }});
    </script>
  </body>
</html>
'''

# Path to the simulated Whisper output (for demo, use a text file)
WHISPER_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'whisper_output.txt')
MERMAID_MMD_PATH = os.path.join(os.path.dirname(__file__), 'diagram.mmd')
MERMAID_HTML_PATH = os.path.join(os.path.dirname(__file__), 'diagram.html')

# Dummy function to convert text to Mermaid code (replace with LLM/your logic)
def text_to_mermaid(text):
    # For demo, just return a static diagram
    return '''flowchart TD\n    A[Christmas] -->|Get money| B(Go shopping)\n    B --> C{Let me think}\n    C -->|One| D[Laptop]\n    C -->|Two| E[iPhone]\n    C -->|Three| F[fa:fa-car Car]'''

# Main loop: watch for changes in whisper_output.txt and update diagram
last_text = None
print("Watching for Whisper output in real time...")
while True:
    try:
        if os.path.exists(WHISPER_OUTPUT_PATH):
            with open(WHISPER_OUTPUT_PATH, 'r') as f:
                text = f.read().strip()
        else:
            text = ''
        if text != last_text:
            print("Detected new input. Generating diagram...")
            mermaid_code = text_to_mermaid(text)
            # Write Mermaid code to .mmd file
            with open(MERMAID_MMD_PATH, 'w') as f:
                f.write(mermaid_code)
            # Write HTML file for live rendering
            with open(MERMAID_HTML_PATH, 'w') as f:
                f.write(MERMAID_HTML_TEMPLATE.format(mermaid_code=mermaid_code))
            print("Diagram updated!")
            last_text = text
        time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        print("Exiting real-time Mermaid renderer.")
        break 