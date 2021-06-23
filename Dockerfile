FROM jinaai/jina:master as base

COPY . ./sentence-transformer/
WORKDIR ./sentence-transformer

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]