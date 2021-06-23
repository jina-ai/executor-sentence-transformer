__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-MY-DUMMY-EXECUTOR',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='This is my dummy executor',
    url='https://github.com/jina-ai/EXECUTOR_REPO_NAME',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.SUB_PACKAGE_NAME.MODULE_NAME'],
    package_dir={'jinahub.SUB_PACKAGE_NAME': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
