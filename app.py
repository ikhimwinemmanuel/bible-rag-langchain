import gradio as gr

from src.rag_qa import answer_with_sources


def _format_sources(sources):
    """Render retrieved passages as a collapsible markdown block."""
    if not sources:
        return ""
    lines = ["\n\n<details><summary>📖 Sources</summary>\n"]
    for s in sources:
        snippet = s["text"].strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        lines.append(f"\n**{s['ref']}** — {snippet}\n")
    lines.append("\n</details>")
    return "".join(lines)


def chat_fn(message, history):
    """Answer the user's message and attach the passages that grounded it."""
    answer, sources = answer_with_sources(message)
    return answer + _format_sources(sources)


demo = gr.ChatInterface(
    fn=chat_fn,
    title=" Bible RAG Assistant",
    description=(
        "Ask questions and get answers grounded only in the Bible text. "
        "Each answer shows the exact passages it was drawn from."
    ),
)

if __name__ == "__main__": # Bind to all interfaces so the app is accessible from Docker and deployment platforms
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
