"""Prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about mining accident reports.\n"
            "Use ONLY the numbered passages provided. Do not use outside knowledge.\n"
            "Rules:\n"
            "1. Every statement must be supported by the passages.\n"
            "2. Cite the passage number in square brackets after each statement, for example [1] or [2][3].\n"
            "3. If the passages do not contain enough information to answer, reply with exactly: "
            + INSUFFICIENT_CONTEXT
            + "\n"
            "4. Be concise and factual. Do not guess causes, dates, names or figures that are not stated.",
        ),
        ("human", "Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You improve search queries for a vector search over mining accident investigation reports. "
            "Rewrite the query using more specific terminology, synonyms and the key entities, "
            "such as equipment, hazard types, locations or mine names. "
            "Return only the rewritten query on a single line.",
        ),
        ("human", "Original question: {question}\nPrevious query: {query}\nRewritten query:"),
    ]
)
