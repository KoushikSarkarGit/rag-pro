"""Advanced RAG patterns: hybrid search with BM25 and Chroma."""

import os
from collections.abc import Iterable
from typing import Any

import bm25s
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import PrivateAttr

load_dotenv()


TECH_DOCS = [
    Document(
        page_content=(
            "Python is a high-level programming language known for its simplicity "
            "and readability. It supports multiple programming paradigms including "
            "procedural, object-oriented, and functional programming. Python is "
            "widely used in web development, data science, artificial intelligence, "
            "and automation."
        ),
        metadata={"topic": "programming", "language": "python", "difficulty": "beginner"},
    ),
    Document(
        page_content=(
            "JavaScript is the language of the web. It runs in browsers and on servers "
            "with Node.js. Modern frameworks like React, Vue, and Angular make building "
            "interactive web applications efficient. JavaScript supports asynchronous "
            "programming with Promises and async/await."
        ),
        metadata={"topic": "programming", "language": "javascript", "difficulty": "intermediate"},
    ),
    Document(
        page_content=(
            "Machine learning is a subset of AI that enables systems to learn from data. "
            "Supervised learning uses labeled data, while unsupervised learning finds "
            "patterns in unlabeled data. Popular ML frameworks include TensorFlow, "
            "PyTorch, and scikit-learn."
        ),
        metadata={"topic": "ai", "subtopic": "machine_learning", "difficulty": "advanced"},
    ),
    Document(
        page_content=(
            "LangChain is a framework for building LLM applications. It provides tools "
            "for prompts, chains, agents, and memory. LangChain supports multiple LLM "
            "providers including OpenAI, Anthropic, and local models."
        ),
        metadata={"topic": "ai", "subtopic": "llm_frameworks", "difficulty": "intermediate"},
    ),
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications "
            "with LLMs. Key features include state management, cycles and loops, "
            "human-in-the-loop workflows, and persistence. LangGraph extends LangChain "
            "for complex agent architectures."
        ),
        metadata={"topic": "ai", "subtopic": "llm_frameworks", "difficulty": "advanced"},
    ),
    Document(
        page_content=(
            "Docker is a platform for containerizing applications. Containers package "
            "code and dependencies together for consistent deployment. Docker Compose "
            "orchestrates multi-container applications. Kubernetes scales Docker "
            "containers in production."
        ),
        metadata={"topic": "devops", "subtopic": "containers", "difficulty": "intermediate"},
    ),
    Document(
        page_content=(
            "PostgreSQL is an advanced open-source relational database. It supports JSON "
            "data types, full-text search, and extensions like pgvector for vector "
            "similarity search. PostgreSQL is ACID compliant and highly extensible."
        ),
        metadata={"topic": "database", "type": "relational", "difficulty": "intermediate"},
    ),
    Document(
        page_content=(
            "Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing "
            "and searching embeddings. They enable semantic similarity search for RAG "
            "applications. Most support metadata filtering and hybrid search combining "
            "keywords with vectors."
        ),
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]


class BM25SRetriever(BaseRetriever):
    """Expose a bm25s index through LangChain's standard retriever interface."""

    docs: list[Document]
    k: int = 4
    _index: Any = PrivateAttr()

    @classmethod
    def from_documents(
        cls, documents: Iterable[Document], *, k: int = 4
    ) -> BM25SRetriever:
        docs = list(documents)
        if not docs:
            raise ValueError("BM25SRetriever requires at least one document.")

        index = bm25s.BM25()
        corpus_tokens = bm25s.tokenize([doc.page_content for doc in docs])
        index.index(corpus_tokens, show_progress=False)

        retriever = cls(docs=docs, k=k)
        retriever._index = index
        return retriever

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_tokens = bm25s.tokenize(query)
        indices, _scores = self._index.retrieve(
            query_tokens,
            k=min(self.k, len(self.docs)),
            show_progress=False,
        )
        return [self.docs[int(index)] for index in indices[0]]


class ReciprocalRankFusionRetriever(BaseRetriever):
    """Combine retriever rankings with weighted reciprocal-rank fusion."""

    retrievers: list[BaseRetriever]
    weights: list[float]
    k: int = 4
    rank_constant: int = 60

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if not self.retrievers:
            raise ValueError("At least one retriever is required for rank fusion.")
        if len(self.retrievers) != len(self.weights):
            raise ValueError("Provide one weight for each retriever.")

        scores: dict[str, float] = {}
        documents_by_content: dict[str, Document] = {}

        for retriever_index, (retriever, weight) in enumerate(
            zip(self.retrievers, self.weights)
        ):
            documents = retriever.invoke(
                query,
                config={
                    "callbacks": run_manager.get_child(
                        tag=f"retriever_{retriever_index + 1}"
                    )
                },
            )
            for rank, document in enumerate(documents, start=1):
                content = document.page_content
                scores[content] = scores.get(content, 0.0) + weight / (
                    rank + self.rank_constant
                )
                documents_by_content.setdefault(content, document)

        ranked_content = sorted(scores, key=scores.get, reverse=True)[: self.k]
        return [documents_by_content[content] for content in ranked_content]


def create_embeddings() -> OpenAIEmbeddings:
    """Create the OpenRouter-backed embeddings client used by Chroma."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
        )

    return OpenAIEmbeddings(
        model="nvidia/nemotron-3-embed-1b:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )


def create_base_vectorstore() -> Chroma:
    """Build a Chroma vector store using the configured OpenRouter embeddings."""
    return Chroma.from_documents(documents=TECH_DOCS, embedding=create_embeddings())


def demo_ensemble_hybrid_search() -> None:
    """Compare keyword, semantic, and reciprocal-rank-fusion results."""
    print("\n" + "=" * 70)
    print("HYBRID SEARCH: BM25S + CHROMA VECTOR SEARCH")
    print("=" * 70)

    bm25_retriever = BM25SRetriever.from_documents(TECH_DOCS, k=3)
    vectorstore = create_base_vectorstore()
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    hybrid_retriever = ReciprocalRankFusionRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
        k=3,
    )

    queries = [
        "PostgreSQL pgvector",
        "What database stores embeddings?",
    ]
    for query in queries:
        print(f"\nQuery: {query}")

        for label, retriever in (
            ("BM25S", bm25_retriever),
            ("Semantic", semantic_retriever),
            ("Hybrid", hybrid_retriever),
        ):
            documents = retriever.invoke(query)
            print(f"  {label} r1: {documents[0].page_content[:200]}...")
            print(f"  {label} r2: {documents[1].page_content[:200]}...")


if __name__ == "__main__":
    demo_ensemble_hybrid_search()
