__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Generator, Dict, List

from jina import Executor, DocumentArray, requests
from sentence_transformers import SentenceTransformer


class ExecutorSentenceTransformer(Executor):
    """
    Encode the Document text into embedding.

    :param embedding_dim: the output dimensionality of the embedding
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-mpnet-base-v2',
        device: str = "cpu",
        default_traversal_paths: List[str] = None,
        default_batch_size=32,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.model = SentenceTransformer(model_name, device=device)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `256`.
        :param parameters: Any additional parameters for the `encode` function.
        """
        for batch in self._get_docs_batch_generator(docs, parameters):
            texts = batch.get_attributes("text")
            embeddings = self.model.encode(texts)
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding

    def _get_docs_batch_generator(self, docs: DocumentArray, parameters: Dict):
        traversal_paths = parameters.get("traversal_paths", self.default_traversal_paths)
        batch_size = parameters.get("batch_size", self.default_batch_size)
        flat_docs = docs.traverse_flat(traversal_paths)
        filtered_docs = DocumentArray(
            [doc for doc in flat_docs if doc is not None and doc.text is not None]
        )
        return _batch_generator(filtered_docs, batch_size)


def _batch_generator(
    data: DocumentArray, batch_size: int
) -> Generator[DocumentArray, None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i: min(i + batch_size, len(data))]
