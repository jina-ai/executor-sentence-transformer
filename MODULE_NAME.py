__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional
import numpy as np

from jina import Executor, DocumentArray, requests


class MyDummyExecutor(Executor):
    """
    Encode the Document blob into embedding.

    :param embedding_dim: the output dimensionality of the embedding
    """

    def __init__(self, embedding_dim: int = 128, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self._dim = embedding_dim

    @requests
    def encode(self, docs: Optional[DocumentArray], **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `256`.
        """
        if not docs:
            return
        for doc in docs:
            assert doc.blob.shape[0] == 256
            doc.embedding = np.random.rand(self._dim)