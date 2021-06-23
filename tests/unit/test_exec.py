__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors import BaseExecutor


def test_exec():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex._dim == 128
