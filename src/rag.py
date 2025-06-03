from typing import List, Dict, Any, Optional

# Global langchain imports
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# Specific modules and local RAG related imports
from llm import LLMClient
from config import ConfigManager
from langchain_ollama import OllamaEmbeddings
from decorators import handle_exception, timing_decorator


class EmbeddingService:
    """Handles document embedding and vector database operations."""

    def __init__(self, embedding_settings: Dict[str, Any]):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = embedding_settings.get("embedding_model", "nomic-embed-text")
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.vector_db = None

    @handle_exception
    @timing_decorator
    def create_vector_db(self, documents: List[Document]) -> DocArrayInMemorySearch:
        """
        Create a vector database from documents.

        Args:
            documents: List of documents to embed

        Returns:
            Vector database containing document embeddings
        """
        self.vector_db = DocArrayInMemorySearch.from_documents(
            documents, self.embeddings
        )
        return self.vector_db

    @handle_exception
    @timing_decorator
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector database.

        Args:
            query: Query string to search for
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database not created. Call create_vector_db() first."
            )

        return self.vector_db.similarity_search(query, k=k)

    @handle_exception
    @timing_decorator
    def get_retriever(self):
        """
        Get a retriever from the vector database.

        Returns:
            Retriever object
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database not created. Call create_vector_db() first."
            )

        return self.vector_db.as_retriever()


class RAGSystem:
    """Retrieval-Augmented Generation system for document QA."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the RAG system.

        Args:
            embedding_model: Name of the embedding model to use
            llm_model: Name of the LLM model to use
            temperature: Temperature parameter for the LLM
        """
        # Load setings
        self.config = config or ConfigManager()
        self.rag_settings = self.config.get("rag", {})

        # Get parameters
        self.filepath = self.rag_settings.get("filepath", {})
        self.documents = self.get_documents(self.filepath)

        # Initialize services
        self.embedding_service = EmbeddingService(self.rag_settings)
        self.llm_service = LLMClient(self.config).infer()
        self.vector_db = self.build_index()

    @handle_exception
    def get_documents(self, filepath: str = None) -> List[Document]:
        """
        Load documents from a CSV file.

        Args:
            filepath: Path to the CSV file (overrides the one set at initialization)
        """
        self.filepath = filepath or self.rag_settings.get("filepath")

        if not self.filepath:
            raise ValueError("No file path provided")

        loader = CSVLoader(self.filepath)
        return loader.load()

    @handle_exception
    @timing_decorator
    def build_index(self) -> DocArrayInMemorySearch:
        """
        Build the vector index from loaded documents.

        Returns:
            Vector database containing document embeddings
        """
        if self.documents is None:
            raise ValueError("No documents loaded. Call load_data() first.")

        return self.embedding_service.create_vector_db(self.documents)

    @handle_exception
    @timing_decorator
    def direct_search_and_answer(self, query: str) -> str:
        """
        Perform a manual search and generate an answer using the LLM.

        Args:
            query: Query string

        Returns:
            LLM response
        """
        if self.vector_db is None:
            raise ValueError("Vector index not built. Call build_index() first.")

        results = self.embedding_service.similarity_search(query)
        context = "".join([doc.page_content for doc in results])
        return self.llm_service.invoke(f"{context} Question: {query}")

    @handle_exception
    @timing_decorator
    def create_qa_chain(
        self, chain_type: str = "stuff", verbose: bool = False
    ) -> RetrievalQA:
        """
        Create a QA chain for more complex retrieval methods.

        Args:
            chain_type: Type of chain to use ("stuff", "map_reduce", "refine", or "map_rerank")
            verbose: Whether to display verbose output

        Returns:
            RetrievalQA chain
        """
        if self.vector_db is None:
            raise ValueError("Vector index not built. Call build_index() first.")

        retriever = self.embedding_service.get_retriever()
        return RetrievalQA.from_chain_type(
            llm=self.llm_service,
            chain_type=chain_type,
            retriever=retriever,
            verbose=verbose,
        )

    @handle_exception
    @timing_decorator
    def create_custom_index(self) -> Any:
        """
        Create a custom vector index using VectorstoreIndexCreator.

        Returns:
            Custom vector index
        """
        if self.documents is None:
            raise ValueError("No documents loaded. Call load_data() first.")

        loader = CSVLoader(self.filepath or self.rag_settings.get("filepath"))
        return VectorstoreIndexCreator(
            vectorstore_cls=DocArrayInMemorySearch,
            embedding=self.embedding_service.embeddings,
        ).from_loaders([loader])

    @handle_exception
    @timing_decorator
    def query_with_chain(
        self, query: str, chain_type: str = "stuff", verbose: bool = False
    ) -> str:
        """
        Query using a specific chain type.

        Args:
            query: Query string
            chain_type: Type of chain to use
            verbose: Whether to display verbose output

        Returns:
            Response from the QA chain
        """
        qa_chain = self.create_qa_chain(chain_type=chain_type, verbose=verbose)
        return qa_chain.run(query)
