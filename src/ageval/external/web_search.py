# pylint: disable=line-too-long
"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Utils for web searching

# Include a RAPID API key in your env.
export RAPID_API_KEY=<>
"""
import copy
import dataclasses
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from absl import app, flags, logging
from bs4 import BeautifulSoup

FLAGS = flags.FLAGS


@dataclasses.dataclass
class Document:
    # Passage text.
    text: str
    # Original web url.
    url: str
    # Title of web url.
    title: str
    # Selected passage.
    passage: Optional[str] = ""
    # Similarity score.
    similarity_score: Optional[float] = None


class WebSearchType(Enum):
    # Rapidapi https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-web-search.
    RapidAPI = "rapidapi"


class PassageSelectorType(Enum):
    Test = "test"


# pylint: disable-next=too-many-instance-attributes
class WebSearch:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        web_search_type: WebSearchType = WebSearchType.RapidAPI,
        passage_selector_type: Optional[PassageSelectorType] = None,
        min_similarity: float = 0.4,
        limit: int = 3,
        passage_max_token_count: Optional[int] = None,
    ):
        """Initializes a web search instance

        Args:
            api_key: A string representing the user's API key for authentication.
            max_retries: Max number of retries.
            web_search_type: Type of web search api.
            passage_selection: The type of passage selector.
            min_similarity: The minimum similarity score to include a document in the results.
            limit: Max limit of urls for web search.
            passage_max_token_count: Set this for passage max token count.
                By default, it is inferred from embedding_ctx_length 254 from text embeddings.
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.web_search_type = web_search_type
        self.passage_selector = None
        self.passage_selector_type = passage_selector_type
        self.min_similarity = min_similarity
        self.limit = limit
        self.passage_max_token_count = passage_max_token_count

    def search(
        self, query: str, *, select_passages: bool = True, extract_url: bool = True
    ) -> Optional[List[Document]]:
        """Fetches top relevant passages from a web search API.

        Args:
            query: Search query.
            select_passages: True to select passages.
            extract_url: True to extract url.

        Returns:
            A list of selected passages.

        Raises:
            ValueError: web_search_type not supported.
        """
        if self.web_search_type != WebSearchType.RapidAPI:
            raise ValueError(f"{self.web_search_type} is not supported")
        try:
            response = self.rapidapi_web_search(query=query)
            logging.debug("Response: %s", response)
            docs = []
            for i in response["data"]:
                if extract_url:
                    text = self.extract_text_from_url(i["url"])
                    if text == "":
                        logging.error("Unable to get text from url %s", i["url"])
                        continue
                else:
                    text = i["snippet"]
                doc = Document(url=i["url"], title=i.get("title", ""), text=text)
                docs.append(doc)
            if not select_passages:
                return docs
            return self.select_passages_from_documents(documents=docs, query=query)
        except requests.HTTPError:
            return None

    def rapidapi_web_search(self, query: str) -> Dict[str, Any]:
        """Fetches data from a web search API.

        This function sends a GET request to the real-time web search API and retrieves
        information. It retries up to max_retries if the request fails
        due to server-side errors.

        Args:
            query: Search query.

        Returns:
            A dictionary containing the JSON response from the API.

        Raises:
            requests.HTTPError: An error occurred while making the request to the API.
        """
        url = "https://real-time-web-search.p.rapidapi.com/search"
        querystring = {"q": query, "limit": self.limit}
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com",
        }
        for _ in range(self.max_retries):  # Retry logic
            response = requests.get(url, headers=headers, params=querystring)
            try:
                response.raise_for_status()  # Raises an HTTPError for bad responses
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code >= 500:  # Retry on server-side errors
                    logging.info("Got an error for web search request %s", str(e))
                    continue
                raise  # Re-raise the exception for client-side errors
        raise requests.HTTPError(
            "Maximum retry attempts exceeded, server is not responding properly."
        )

    def extract_text_from_url(self, url: str) -> str:
        """Extracts text from a web url.

        Args:
            url: A web url.

        Returns:
            a string of web raw text with html tag removed.
        """

        for _ in range(self.max_retries):  # Retry logic
            response = requests.get(url)
            try:
                response.raise_for_status()  # Raises an HTTPError for bad responses
                # Parse the HTML content of the page using BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract text from the parsed HTML (removing all HTML tags)
                text = soup.get_text(separator=" ", strip=True)

                return text
            except requests.exceptions.HTTPError as e:
                if response.status_code >= 500:  # Retry on server-side errors
                    logging.info("Got an error for url request %s", str(e))
                    continue
                return ""
        return ""

    def select_passages_from_documents(
        self, documents: List[Document], query: str
    ) -> Union[None, List[Document]]:
        """Select relevant passages from document text.

        Args:
            documents: A list of web documents.
            query: A string of search query.

        Returns:
            A list of selected relevant passages in a tuple.
            The first element is the index of documents.
            The second element is selected passage text.
            None is return if passage_selector is not initialized.
        """
        if (
            self.passage_selector is None
        ):
            return None
        tokenizer = self.passage_selector.tokenizer

        chunk_texts = []
        chunk_embeddings = []
        chunk_indices = []
        max_token_count = (
            self.passage_max_token_count or self.passage_selector.embedding_ctx_length
        )
        for document_idx, document in enumerate(documents):
            tokens = tokenizer.encode(document.text, add_special_tokens=False)
            for i in range(0, len(tokens), max_token_count):
                chunk_text = tokenizer.decode(tokens[i : i + max_token_count])
                chunk_texts.append(chunk_text)
                chunk_indices.append(document_idx)
        chunk_texts = self.passage_selector._remove_unexpected_character(chunk_texts)
        logging.debug("chunk_texts: %s", chunk_texts)
        for idx, chunk_text in enumerate(chunk_texts):
            doc_emb = self.passage_selector._call_text_embedding([chunk_text])
            chunk_embeddings.append(doc_emb[0])
        logging.debug("chunk_embeddings: %s", chunk_embeddings)
        query_emb = self.passage_selector.embed_query(query)
        # Convert lists to numpy arrays
        query_emb = np.array(query_emb)
        doc_embs = np.array(chunk_embeddings)
        indexed_similarities = rank_documents(
            query_emb=query_emb, doc_embs=doc_embs, min_similarity=self.min_similarity
        )
        indexed_similarities = indexed_similarities[: self.limit]
        top_results = []
        for idx, score in indexed_similarities:
            doc: Document = copy.deepcopy(documents[chunk_indices[idx]])
            doc.passage = chunk_texts[idx]
            doc.similarity_score = score
            top_results.append(doc)
        return top_results


def rank_documents(
    query_emb: np.ndarray, doc_embs: np.ndarray, min_similarity: float = 0.4
) -> List[tuple[int, float]]:
    """Ranks documents by similarity to the query embedding using the dot product, with a cutoff.

    Computes the dot product between the query embedding and each document embedding.
    Only documents with a similarity score above `min_similarity` are considered.

    Args:
      query_emb: The embedding vector of the query.
      doc_embs: A list of embedding vectors for documents.
      min_similarity: The minimum similarity score to include a document in the results.

    Returns:
      A list of indices, each containing the index of a document and its similarity score,
      sorted by similarity in descending order.
    """
    # Calculate dot products
    similarities = np.dot(doc_embs, query_emb)

    # Filter indices and similarities by the cutoff value
    valid_indices = np.where(similarities >= min_similarity)[0]
    valid_similarities = similarities[valid_indices]

    # Create tuples of (index, similarity) and sort them by similarity in descending order
    indexed_similarities = list(zip(valid_indices, valid_similarities))
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)

    return indexed_similarities


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    search_api_key = FLAGS.search_api_key
    if search_api_key is None and "RAPID_API_KEY" in os.environ:
        search_api_key = os.environ["RAPID_API_KEY"]
    if search_api_key is None:
        raise ValueError("Please set api key.")
    web_search = WebSearch(
        api_key=search_api_key,
        min_similarity=FLAGS.min_similarity,
        limit=FLAGS.limit,
        passage_max_token_count=FLAGS.passage_max_token_count,
    )
    docs = web_search.search(query=FLAGS.query, select_passages=FLAGS.select_passages)
    for idx, doc in enumerate(docs):
        logging.info("Rank %d. Document: %s", idx + 1, doc)


def web_search_flags():
    flags.DEFINE_string(
        "query",
        default=None,
        required=True,
        help="""Search query""",
    )
    flags.DEFINE_string(
        "search_api_key",
        default=None,
        required=False,
        help="""Search API key""",
    )
    flags.DEFINE_boolean(
        "select_passages",
        default=False,
        help="""True to run passage selector.""",
    )
    flags.DEFINE_bool(
        "debug",
        default=False,
        help="True to enable debug.",
        required=False,
    )
    flags.DEFINE_float(
        "min_similarity",
        default=0.4,
        help="Similarity score to gate the query/passage similarity",
        required=False,
    )
    flags.DEFINE_integer(
        "limit",
        default=3,
        help="Max number of passages and urls",
        required=False,
    )

    flags.DEFINE_integer(
        "passage_max_token_count",
        default=None,
        help="Max token count for chunking a passage.",
        required=False,
    )


if __name__ == "__main__":
    web_search_flags()
    app.run(main)
