pip install -U openai pypdf
pip install gradio openai pypdf python-docx
pip install gradio requests pypdf python-docx
pip install gradio requests pypdf python-docx rank-bm25
pip install gradio requests pypdf python-docx ddgs beautifulsoup4
pip install duckduckgo-search
export DEEPSEEK_API_KEY="yourAPIkeyhere😄"
python RLM.py
MAX_FILE_MB=500 MAX_TOTAL_CHARS=50000000 python RLM.py
++++++++++++++++++++++++++++++ENV activations+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

python -m venv venv
source /Users/htet/Desktop/X-LLM/XcodesX/venv/bin/activate

++++++++++++++++++++++++++++++PIPS+++++++++++++++++++++++++++++++++++++++++++++++++++++++++


pip install openai pypdf
pip install -U openai pypdf
pip install openai pypdf python-docx
pip install textract

++++++++++++++++++++++++++++++ARC+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

User Query
   ↓
Planner LLM
   ↓
Structured Action JSON
   ↓
Tool Executor
   ├─ search_chunks
   ├─ get_page
   ├─ regex_find
   ├─ summarize_chunk
   └─ recursive_child_run
   ↓
Working Memory / Evidence Store
   ↓
Final Synthesizer
   ↓
Answer + citations + trace

++++++++++++++++++++++++++++++api+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

export DEEPSEEK_API_KEY="yourAPIkeyhere😄"
