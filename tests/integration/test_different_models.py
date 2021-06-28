__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Callable

import pytest

from jina import DocumentArray

from jinahub.text.encoders.sentence_encoder import TransformerSentenceEncoder


MODELS_TO_TEST = [
    'paraphrase-MiniLM-L6-v2',
    'paraphrase-MiniLM-L3-v2',
    'average_word_embeddings_komninos',
    'average_word_embeddings_glove.6B.300d'
]


@pytest.mark.parametrize(
    'model_name', MODELS_TO_TEST
)
def test_load_torch_models(model_name: str, data_generator: Callable):
    encoder = TransformerSentenceEncoder(model_name=model_name)

    docs = DocumentArray([doc for doc in data_generator()])
    encoder.encode(
        docs=docs,
        parameters={}
    )

    for doc in docs:
        assert doc.embedding is not None
