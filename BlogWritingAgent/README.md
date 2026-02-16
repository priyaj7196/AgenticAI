# Blog Writer — AI-Powered Technical Blog Generator

A Streamlit application that provides an intelligent blog-writing agent built with **LangGraph**. It plans, researches, writes, and illustrates technical articles from a single topic prompt—combining LLMs, web search, and AI image generation into a cohesive multi-agent pipeline.

![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)

---

## What It Does

Give it a topic—e.g. *"Positional Encoding in Transformer Architecture"*—and it will:

---

## ⚠️ Known Limitations

| Issue | Status |
|-------|--------|
| **Gemini Image Generation** | Requires `GOOGLE_API_KEY` and `google-genai` installed in the correct Python environment. Free-tier quota may limit image generation. |
| **Tavily Research** | Requires `TAVILY_API_KEY`. Without it, research mode falls back to empty evidence. |
| **Image Fallback** | If image generation fails (quota, safety, SDK), the blog outputs with a placeholder block instead of the diagram. |

---

## Features

- **Multi-Mode Routing**: Automatically decides whether the topic needs web research—evergreen, hybrid, or news roundup
- **Adaptive Research**: Uses [Tavily](https://tavily.com) for web search when needed; filters by recency for news topics
- **Citation Grounding**: Cites provided evidence URLs; avoids unsupported claims in open-book mode
- **Parallel Section Writing**: Drafts sections concurrently for faster runs
- **AI-Generated Diagrams**: Up to 3 technical diagrams per blog via [Gemini 2.5 Flash Image](https://ai.google.dev/gemini-api/docs/image-generation)
- **Graceful Fallbacks**: If image generation fails, the blog still outputs with a descriptive placeholder
- **Past Blogs Browser**: Load and preview previously generated articles from the sidebar
- **Export Options**: Download Markdown only or a ZIP bundle (Markdown + images)
- **Streamlit UI**: Topic input, as-of date, live preview, and download controls

---

## How It Works

### Pipeline Overview

```
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌────────┐     ┌─────────┐
│ Router  │────▶│ Research │────▶│ Orchestrator│────▶│ Workers│────▶│ Reducer │
└─────────┘     └──────────┘     └─────────────┘     └────────┘     └─────────┘
     │                │                  │                │              │
     ▼                ▼                  ▼                ▼              ▼
 closed_book     Tavily Search      Plan (5–9 tasks)   Parallel      Merge + Images
 hybrid          Evidence pack      Structured output  Section write  Gemini diagrams
 open_book       Recency filter     Citations/Code     Per task       Markdown output
```

### Node Descriptions

| Node | Description |
|------|-------------|
| **Router** | Chooses `closed_book` (evergreen), `hybrid` (needs examples), or `open_book` (news/roundup) |
| **Research** | Queries Tavily for sources; filters by recency for news; produces evidence pack |
| **Orchestrator** | Produces a structured `Plan` with 5–9 tasks, bullets, word targets, and citation flags |
| **Workers** | Fan-out to write sections in parallel, each grounded in evidence when required |
| **Reducer** | Merges sections, decides if images help, generates diagrams via Gemini, writes final `.md` |

---

## Configuration System

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Planning and writing (GPT-4.1-mini) |
| `TAVILY_API_KEY` | For research | Web search; omit for closed-book only |
| `GOOGLE_API_KEY` | For images | AI-generated diagrams via Gemini |

Create a `.env` file in the project root or in `LangGraph/BlogWriter`. The app loads `.env` automatically.

### Supported Modes

| Mode | Research | Use Case |
|------|----------|----------|
| `closed_book` | ❌ No | Evergreen concepts (e.g., "Self-Attention in Transformers") |
| `hybrid` | ✅ Yes | Evergreen + up-to-date examples/tools/models |
| `open_book` | ✅ Yes | News roundups, "latest" topics, pricing, policy |

---

## Quick Start

### 1. Setup

```bash
cd LangGraph/BlogWriter
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
OPENAI_API_KEY=sk-...          # Required for planning & writing
TAVILY_API_KEY=tvly-...        # Required for web research
GOOGLE_API_KEY=...             # Required for AI-generated images
```

### 3. Run the App

```bash
streamlit run frontend.py
```

Enter a topic, pick an as-of date, and click **Generate Blog**. The agent streams progress and produces a Markdown file with optional images in the `images/` folder.

---

## Project Structure

| File | Description |
|------|-------------|
| `frontend.py` | Streamlit UI — topic input, past blogs, live preview, download |
| `backend.py` | LangGraph pipeline — router, research, orchestrator, workers, reducer |
| `bwa_backend.py` | Shim for frontend import |
| `BlogWriterWithResearchAndImages.ipynb` | Full notebook (research + images) |
| `BlogWriterWithResearch.ipynb` | Notebook with research, no images |
| `BasicBlogWriter.ipynb` | Minimal notebook, no research or images |
| `images/` | Generated diagrams (created at runtime) |
| `*.md` | Example output blogs |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | OpenAI GPT-4.1-mini (planning & writing) |
| Web search | [Tavily](https://tavily.com) via LangChain |
| Image generation | [Google Gemini 2.5 Flash Image](https://ai.google.dev/gemini-api/docs/image-generation) |
| UI | [Streamlit](https://streamlit.io) |

---

## Example Output

Given the topic *"Understanding Self-Attention in Transformer Architecture"*, the agent produces a structured blog with:

- Introduction to self-attention
- Query, key, value vectors
- Scaled dot-product attention
- Multi-head attention
- Code sketches and math
- Optional diagrams (e.g. QKV flow, attention heads)

Output is saved as `Understanding_Self-Attention_in_Transformer_Architecture.md` with images in `images/`.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned versions

---

## License

MIT
