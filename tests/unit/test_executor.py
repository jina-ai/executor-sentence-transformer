from pathlib import Path
import numpy as np
import pytest
from jina import Executor
from sentence_encoder import TransformerSentenceEncoder
from docarray import DocList
from docarray.documents import TextDoc

_EMBEDDING_DIM = 384


@pytest.fixture(scope='session')
def basic_encoder() -> TransformerSentenceEncoder:
    return TransformerSentenceEncoder()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.__class__.__name__ == 'TransformerSentenceEncoder'


def test_encoding_cpu():
    enc = TransformerSentenceEncoder(device='cpu')
    input_data = DocList[TextDoc]([TextDoc(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_encoding_gpu():
    enc = TransformerSentenceEncoder(device='cuda')
    input_data = DocList[TextDoc]([TextDoc(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    'model_name, emb_dim',
    [
        ('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 384),
        ('sentence-transformers/msmarco-distilbert-base-tas-b', 768),
        ('distilbert-base-uncased', 768),
    ],
)
def test_models(model_name: str, emb_dim: int):
    encoder = TransformerSentenceEncoder(model_name)
    input_data = DocList[TextDoc]([TextDoc(text='hello world')])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (emb_dim,)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: TransformerSentenceEncoder, batch_size: int):
    docs = DocList[TextDoc]([TextDoc(text='hello there') for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


def test_quality_embeddings(basic_encoder: TransformerSentenceEncoder):
    docs = DocList[TextDoc](
        [
            TextDoc(id='A', text='a furry animal that with a long tail'),
            TextDoc(id='B', text='a domesticated mammal with four legs'),
            TextDoc(id='C', text='a type of aircraft that uses rotating wings'),
            TextDoc(id='D', text='flying vehicle that has fixed wings and engines'),
        ]
    )

    basic_encoder.encode(docs, {})

    # assert semantic meaning is captured in the encoding
    matches = ['B', 'A', 'D', 'C']

    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    # Compute pairwise cosine similarities
    num_docs = len(docs)
    similarity_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        for j in range(num_docs):
            similarity_matrix[i][j] = cosine_similarity(docs[i].embedding, docs[j].embedding)

    def get_most_similar(idx, matrix):
        row = matrix[idx]
        # Setting similarity with self to -1 to exclude it from being the max
        row[idx] = -1
        return np.argmax(row)

    for idx, doc in enumerate(docs):
        assert docs[get_most_similar(idx, similarity_matrix)].id == matches[idx]

