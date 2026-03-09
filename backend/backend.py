from os import getenv
from chromadb import PersistentClient
from llama_index.core.llms import ChatMessage
from chromadb.config import Settings as ChromaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Canadian Tax Advisor", host="0.0.0.0", port=8000)


def get_retriever():
    chroma_client = PersistentClient(
        path="./chroma_db", settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(name="chunks")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = NVIDIAEmbedding(
        api_key=getenv("NVIDIA_API_KEY"),
        model=getenv("NVIDIA_EMBEDDING_MODEL"),
    )
    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return vector_store_index.as_retriever(similarity_top_k=8)


@mcp.tool()
def query(question: str) -> list[str]:
    """Retrieve the most relevant Canadian tax knowledge chunks for a given question.
    Use this tool when the user asks any question related to Canadian taxes, including but not limited to:
    income tax, capital gains, property sales, rental income, self-employment, deductions, credits,
    tax filing, CRA rules, and tax implications of life events.
    Input: a natural language question in English or Chinese.
    Output: a list of relevant text chunks from the tax knowledge base, ranked by relevance.
    """
    keywords = extract_keywords(question)
    enriched_question = f"{question}\n\nRelated keywords: {keywords}"
    nodes = retriever.retrieve(enriched_question)
    return [node.get_content() for node in nodes]


KEYWORDS_SYSTEM_PROMPT = """You are a tax terminology extraction and prediction expert with deep knowledge of Canadian tax systems.

Your task:
- Extract keywords directly from the user's input that describe their specific case (people, transactions, assets, events, relationships, amounts, dates, etc.)
- Expand each keyword with synonyms, more precise/professional equivalents, and related tax terminology
- Include relevant tax concepts that apply to the user's described situation even if not mentioned
- Provide all keywords in both English and Chinese, mixed together in a single list

Output format:
Return a single flat comma-separated list of keywords, e.g.: keyword1, keyword2, 关键词, keyword3, 关键词, ...

Rules:
- Cover both the user's case descriptors (e.g. "property sale", "rental income", "self-employed") and the tax concepts they trigger (e.g. "capital gains", "CCA", "T2125")
- Include synonyms and more professional/precise alternatives for terms used by the user
- Include both English and Chinese terms mixed in the same list
- Do not use any headers, bullet points, numbering, or extra explanation — only the comma-separated list
- Be concise and precise"""


def extract_keywords(text: str) -> str:
    language_model = NVIDIA(
        api_key=getenv("NVIDIA_API_KEY"),
        model=getenv("NVIDIA_LLM_MODEL"),
        is_chat_model=True,
    )
    response = language_model.chat(
        [
            ChatMessage(role="system", content=KEYWORDS_SYSTEM_PROMPT),
            ChatMessage(role="user", content=text),
        ]
    )
    return response.message.content


retriever = get_retriever()

if __name__ == "__main__":
    # question = "老师介绍了哪些合理避税的方法?"
    # print(query(question))
    mcp.run("streamable-http")
