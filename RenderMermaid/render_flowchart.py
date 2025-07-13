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

WHISPER_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'whisper_output.txt')
MERMAID_HTML_PATH = os.path.join(os.path.dirname(__file__), 'diagram.html')

# Read the formatted Mermaid code from whisper_output.txt
def read_mermaid_code():
    if os.path.exists(WHISPER_OUTPUT_PATH):
        with open(WHISPER_OUTPUT_PATH, 'r') as f:
            text = f.read().strip()
        # Extract the code block if present
        if text.startswith('```mermaid'):
            lines = text.splitlines()
            # Remove the first and last lines (```mermaid and ```)
            code = '\n'.join(lines[1:-1]).strip()
            return code
        else:
            return text
    return ''

# mermaid_code = read_mermaid_code()

# # Write HTML file for live rendering
# with open(MERMAID_HTML_PATH, 'w') as f:
#     f.write(MERMAID_HTML_TEMPLATE.format(mermaid_code=mermaid_code))

# print("Diagram rendered to diagram.html!")

def render_mermaid_html():
    mermaid_code = read_mermaid_code()
    with open(MERMAID_HTML_PATH, 'w') as f:
        f.write(MERMAID_HTML_TEMPLATE.format(mermaid_code=mermaid_code))
    print("Diagram rendered to diagram.html!")

if __name__ == "__main__":
    render_mermaid_html()
