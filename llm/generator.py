from llm.gemini_client import get_gemini_model

def generate_structured_answer(query, documents):

    model = get_gemini_model()

    # Build context with citations
    context = ""
    for i, doc in enumerate(documents):
        citation = f"[Source {i+1} - Page {doc.metadata.get('page', 'N/A')}]"
        context += f"{citation}\n{doc.page_content}\n\n"

    prompt = f"""
You are a financial analyst assistant.

Using ONLY the provided sources, answer the question in a structured format:

Question:
{query}

Instructions:
- Provide a concise executive summary.
- Use bullet points for key financial metrics.
- Include growth percentages where available.
- Cite sources inline like this: (Source 1), (Source 2).
- Do not hallucinate beyond the given data.

Sources:
{context}
"""

    response = model.invoke(prompt)

    return response.content