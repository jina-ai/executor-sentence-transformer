__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import pytest
import torch
from jina import DocumentArray, Document

from jinahub.text.encoders.sentence_encoder import TransformerSentenceEncoder


def test_encoding_cpu():
    enc = TransformerSentenceEncoder(device="cpu")
    input_data = DocumentArray([Document(text="hello world")])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (768,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is needed for this test")
def test_encoding_gpu():
    enc = TransformerSentenceEncoder(device="cuda")
    input_data = DocumentArray([Document(text="hello world")])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (768,)


def test_encodes_semantic_meaning():
    sentences = dict()
    sentences["A"] = "Hello, my name is Michael."
    sentences["B"] = "Today we are going to Disney World."
    sentences["C"] = "There are animals on the road"
    sentences["D"] = "A dog is running down the road"

    encoder = TransformerSentenceEncoder()

    embeddings = {}
    for id_, sentence in sentences.items():
        docs = DocumentArray([Document(text=sentence)])
        encoder.encode(docs, parameters={})
        embeddings[id_] = docs[0].embedding

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist("C", "D")
    assert small_distance < dist("C", "B")
    assert small_distance < dist("C", "A")
    assert small_distance < dist("B", "A")
