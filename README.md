# ✨ Executor Sentence Encoder 

**ExecutorSentenceTransformer** wraps the [Sentence Transformer](https://www.sbert.net/docs)
library into an `Jina` executor. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

The [dependencies](requirements.txt) for this executor can be installed using `pip install -r requirements.txt`.
The test suite has additional [requirements](tests/requirements.txt).

## 🚀 Usages
1. Install the `executor-sentence-transformer` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-image-torch-encoder.git
	```

1. Use `executor-sentence-transformer` in your code

```python
from jina import Flow
from jinahub.text.encoders.sentence_encoder import ExecutorSentenceTransformer
f = Flow().add(uses=ExecutorSentenceTransformer)
```

### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-sentence-transformer.git
	cd executor-sentence-transformer
	docker build -t executor-sentence-transformer .
	```

1. Use `executor-sentence-transformer` in your codes

	```python
	from jina import Flow

	f = Flow().add(uses='docker://executor-sentence-transformer:latest')
	```


## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='docker://executor-sentence-transformer:latest')

with f:
    resp = f.post(on='foo', inputs=Document(), return_resutls=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `text` sentences.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (depends on the used model) with `dtype=nfloat32`.


## 🔍️ Reference
- [Sentence Transformer Library](https://www.sbert.net/docs)

