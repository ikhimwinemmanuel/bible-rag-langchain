from langchain_core.prompts import PromptTemplate

# Single source of truth for the refusal text. Both the retrieval gate (in
# rag_qa) and the LLM (via the system prompt below) must emit EXACTLY this
# string so refusals are detectable and consistent. Plain ASCII apostrophe.
NO_CONTEXT_MESSAGE = "I couldn't find a grounded answer in the retrieved passages."

SYSTEM_PROMPT = f"""
You are a Bible research assistant.

Answer Bible-related questions using ONLY the provided context:
- Answer ONLY using the provided context.
- Do not use outside knowledge.
- Do not hallucinate.
- Rely on similar meaning in the provided Bible passages.
- Each passage is labelled with its reference, e.g. [Genesis 1:1-5].
- Always cite the reference of the passage you used, e.g. [Genesis 1:3].

If the answer is not grounded in the context, reply with EXACTLY this sentence
and nothing else:
"{NO_CONTEXT_MESSAGE}"
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context (retrieved Bible passages):
{context}

Instructions:
- Use ONLY the context above.
- Provide short, accurate answers.
- Cite the reference of each passage you use, e.g. [Genesis 1:3].
"""

BIBLE_RAG_PROMPT = PromptTemplate.from_template(USER_PROMPT_TEMPLATE)
