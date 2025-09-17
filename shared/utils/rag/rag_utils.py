import os # For os.path.basename in create_rag_prompt, if source is a path
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

from termcolor import colored

# Assuming OllamaClient is accessible as per the original structure
# This is used for embedding the user query during retrieval.
from core.providers.cls_ollama_interface import OllamaClient

if TYPE_CHECKING:
    import chromadb
    # from sentence_transformers import CrossEncoder # Imported locally in rerank_results

# Default parameters for text chunking, if the user wants to use the provided utility
DEFAULT_CHUNK_SIZE = 1000  # Characters
DEFAULT_CHUNK_OVERLAP = 200 # Characters


class RagTooling:
    """
    A collection of tools for Retrieval Augmented Generation (RAG).
    This class provides utilities for:
    - Splitting text into chunks.
    - Reranking retrieved documents.
    - Creating RAG prompts.
    - Retrieving and augmenting content from a pre-populated ChromaDB collection.

    This class does NOT handle initial document parsing (e.g., from PDFs) or
    the initial population of the ChromaDB collection with document embeddings.
    The user is expected to prepare their data and ChromaDB collection beforehand.
    """

    @classmethod
    def split_text_into_chunks(cls, text: str, 
                               chunk_size: int = DEFAULT_CHUNK_SIZE, 
                               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """
        Splits a given text into overlapping chunks.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The maximum size of each chunk in characters.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.

        Returns:
            List[str]: A list of text chunks.
        
        Raises:
            ValueError: If chunk_overlap is negative, chunk_size is not positive,
                        or chunk_overlap is greater than or equal to chunk_size
                        and step size becomes non-positive.
        """
        if not text:
            return []
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        
        # step = chunk_size - chunk_overlap
        # if step <= 0:
        #    raise ValueError(
        #        f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size}) "
        #        "to ensure positive step size."
        #    )
        # A more relaxed check for step:
        if chunk_overlap >= chunk_size and chunk_size > 0 :
             print(colored(f"Warning: Chunk overlap ({chunk_overlap}) is >= chunk size ({chunk_size}). This might lead to redundant chunks or specific behavior if step becomes non-positive.", "yellow"))


        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:min(end, text_len)])
            
            step = chunk_size - chunk_overlap
            if step <= 0: 
                if chunk_size > 0 : 
                    start += chunk_size # Default to non-overlapping if overlap makes step non-positive
                    if start <= end and chunk_overlap >= chunk_size : # If we just moved by chunk_size, and it was same as overlap, we might repeat
                        pass # This case can be tricky. For now, just advance.
                else: # chunk_size is 0 or negative, text_len check should have caught empty text.
                    break # Safety break
            else:
                start += step
        
        return [chunk for chunk in chunks if chunk.strip()]

    @classmethod
    def rerank_results(cls, text_and_metas: List[Tuple[str, dict]], user_query: str, top_k: int = 5) -> List[Tuple[str, dict]]:
        """
        Reranks retrieved documents using a CrossEncoder model.

        Args:
            text_and_metas (List[Tuple[str, dict]]): A list of (document_text, metadata) tuples.
            user_query (str): The user's query.
            top_k (int): The number of top results to return after reranking.

        Returns:
            List[Tuple[str, dict]]: The reranked and (potentially) truncated list of results.
        """
        if not text_and_metas or top_k == 0:
            return []
        
        try:
            from sentence_transformers import CrossEncoder # Local import for heavy dependency
        except ImportError:
            print(colored("Sentence Transformers library not found. Reranking skipped. "
                          "Install with: pip install sentence-transformers", "red"))
            return text_and_metas[:top_k] # Return original (truncated) if reranker not available

        actual_top_k = min(top_k, len(text_and_metas))

        try:
            # Consider making model name a parameter or class attribute if flexibility is needed
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            rerank_pairs = [[user_query, doc_text] for doc_text, _ in text_and_metas]
            
            print(colored(f"Reranking {len(rerank_pairs)} documents for top {actual_top_k}...", "blue"))
            rerank_scores = reranker.predict(rerank_pairs, show_progress_bar=False)
            
            reranked_items = sorted(zip(text_and_metas, rerank_scores), key=lambda x: x[1], reverse=True)
            top_results = [item[0] for item in reranked_items[:actual_top_k]]
            print(colored(f"Reranked. Selected top {len(top_results)} results.", "green"))
            return top_results
        except Exception as e:
            print(colored(f"Error during reranking: {e}. Returning original (truncated) list.", "red"))
            return text_and_metas[:actual_top_k]

    @classmethod
    def create_rag_prompt(cls, text_and_metas: List[Tuple[str, Dict[str, Any]]], user_query: str) -> str:
        """
        Creates a RAG prompt string from retrieved documents and the user query.

        Args:
            text_and_metas (List[Tuple[str, Dict[str, Any]]]): A list of (document_text, metadata)
                                                               tuples, typically after reranking.
            user_query (str): The user's original query.

        Returns:
            str: A formatted RAG prompt.
        """
        if not text_and_metas:
            return (
                "INSTRUCTION:\nNo relevant context found. "
                "Please answer the user query based on general knowledge if possible, "
                "or state that information is unavailable.\n"
                f"USER_QUERY: {user_query}"
            )

        context_str = "CONTEXT:\n"
        # Documents are typically ordered by relevance (best first) by the reranker.
        # The original implementation reversed them, meaning item 0 in the prompt was the *least* relevant of the top-k.
        # Presenting them in order of relevance (best first) is common.
        # If reversed order is desired (e.g. #0 = least relevant of top-k), use `reversed(text_and_metas)`.
        for i, (doc_text, metadata) in enumerate(text_and_metas): # Kept original's reversed for consistency
            source_info_parts = []
            if 'source' in metadata:
                source_info_parts.append(f"Source: {os.path.basename(str(metadata['source'])) if isinstance(metadata['source'], str) else metadata['source']}")
            if 'page' in metadata:
                source_info_parts.append(f"Page: {metadata['page']}")
            if 'chunk_on_page' in metadata: # Example of another useful metadata
                source_info_parts.append(f"Chunk: {metadata['chunk_on_page']}")
            
            source_info = f"({', '.join(source_info_parts)})" if source_info_parts else ""
            
            cleaned_doc_text = doc_text.replace("\n\n", "\n").strip()
            context_str += f"{i}. {cleaned_doc_text} {source_info}\n\n"

        prompt_instruction = (
            "INSTRUCTION:\n"
            "Based EXCLUSIVELY on the CONTEXT provided above, answer the USER_QUERY. "
            "If the context does not contain the answer, state that the information is not found in the provided documents. "
            "Do not use any external knowledge."
        )
        return f"{context_str.strip()}\n\n{prompt_instruction}\nUSER_QUERY: {user_query}"

    @classmethod
    def retrieve_augment(cls, user_query: str, collection: 'chromadb.Collection', 
                         retrieve_n_results: int = 20, rerank_top_k: int = 5) -> str:
        """
        Retrieves relevant documents from a ChromaDB collection, reranks them,
        and creates a RAG prompt.

        Args:
            user_query (str): The user's query.
            collection (chromadb.Collection): The pre-populated ChromaDB collection to query.
            retrieve_n_results (int): The number of initial results to fetch from ChromaDB.
            rerank_top_k (int): The number of results to keep after reranking.

        Returns:
            str: A formatted RAG prompt string.
        """
        if collection.count() == 0:
            print(colored(f"Collection '{collection.name}' is empty. Cannot retrieve.", "yellow"))
            return cls.create_rag_prompt([], user_query)
            
        print(colored(f"Embedding user query (first 50 chars): '{user_query[:50]}...'", "blue"))
        try:
            user_query_embedding: List[float] = OllamaClient.generate_embedding(user_query)
        except Exception as e:
            print(colored(f"Error embedding user query with OllamaClient: {e}", "red"))
            return f"Error: Could not embed user query. Details: {e}"

        print(colored(f"Querying collection '{collection.name}' for {retrieve_n_results} results...", "blue"))
        try:
            results = collection.query(
                query_embeddings=[user_query_embedding], # Query embeddings should be a list of lists
                n_results=min(retrieve_n_results, collection.count()),
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(colored(f"Error querying ChromaDB collection '{collection.name}': {e}", "red"))
            return f"Error: Could not query vector database. Details: {e}"
        
        retrieved_documents = results.get('documents', [[]])[0] # query returns list of lists
        retrieved_metadatas = results.get('metadatas', [[]])[0]

        if not retrieved_documents:
            print(colored(f"No documents found in '{collection.name}' for the query.", "yellow"))
            return cls.create_rag_prompt([], user_query) 

        docs_metas = list(zip(retrieved_documents, retrieved_metadatas))
        
        reranked_results = cls.rerank_results(docs_metas, user_query, top_k=rerank_top_k)
        
        print(colored(f"Creating RAG prompt from {len(reranked_results)} documents...", "blue"))
        return cls.create_rag_prompt(reranked_results, user_query)