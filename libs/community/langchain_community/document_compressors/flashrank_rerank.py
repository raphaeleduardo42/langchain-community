from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, cast

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, model_validator

if TYPE_CHECKING:
    from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        pass

DEFAULT_MODEL_NAME = "ms-marco-MultiBERT-L-12"


class FlashrankRerank(BaseDocumentCompressor):
    """Document compressor using Flashrank interface."""

    client: Any = None
    """Custom Flashrank client to use for compressing documents."""
    model: str = DEFAULT_MODEL_NAME
    """Model to use for reranking if `client` not defined."""
    top_n: int = 3
    """Number of documents to return."""
    score_threshold: float = 0.0
    """Minimum relevance threshold to return."""
    prefix_metadata: str = ""
    """Prefix for flashrank_rerank metadata keys."""

    filter_metadata_keys: Optional[Set[str]] = None
    """Metadata keys to use when compressing. Input `Document` metadata is preserved.

    If `None` all keys are used.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        """Validate python package existance in environment."""
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError(
                "Could not import flashrank python package. "
                "Please install it with `pip install flashrank`."
            )
        if (client := values.get("client")) and not isinstance(client, Ranker):
            raise ValueError(
                f"Client Ranker expected from type `flashranker.Ranker.Ranker`. "
                f"Defined: {type(client)}."
            )
        return values

    def get_client(self) -> Ranker:
        if self.client:
            return self.client
        else:
            return Ranker(model_name=self.model)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        client = self.get_client()
        passages: List[Dict[str, Any]] = []

        for i, doc in enumerate(documents):
            passage = {"id": i, "text": doc.page_content}
            if self.filter_metadata_keys:
                passage["meta"] = {
                    key: doc.metadata[key]
                    for key in self.filter_metadata_keys
                    if key in doc.metadata.keys()
                }
            else:
                passage["meta"] = doc.metadata
            passages.append(passage)

        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_response = client.rerank(rerank_request)[: self.top_n]
        final_results = []

        for r in rerank_response:
            if r["score"] >= self.score_threshold:
                _id: int = r["id"]
                doc = Document(
                    page_content=r["text"],
                    metadata={
                        self.prefix_metadata + "id": _id,
                        self.prefix_metadata + "relevance_score": r["score"],
                        **documents[_id].metadata,
                    },
                )
                final_results.append(doc)
        return final_results
