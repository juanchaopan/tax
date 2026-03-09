from os import getenv
from jsonlines import open
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings as LlamaIndexSettings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

LlamaIndexSettings.embed_model = NVIDIAEmbedding(
    api_key=getenv("NVIDIA_API_KEY"),
    model=getenv("NVIDIA_EMBEDDING_MODEL"),
)


def load_documents(documents_path: str = "documents.jsonl") -> list[Document]:
    documents: list[Document] = []
    with open(documents_path, "r") as reader:
        for obj in reader:
            doc = Document(doc_id=obj["file"], text=obj["text"])
            documents.append(doc)
    return documents


def build_index(chroma_client: PersistentClient, documents: list[Document]) -> None:
    collection_name = "chunks"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    transformations = [SentenceSplitter(chunk_size=1000, chunk_overlap=200)]
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=transformations,
        show_progress=True,
    )


if __name__ == "__main__":
    documents = load_documents("documents.jsonl")
    chroma_client = PersistentClient(
        path="./chroma_db", settings=ChromaSettings(anonymized_telemetry=False)
    )
    build_index(chroma_client, documents)
    pass
