# 📝 Guidelines for Developing Executors

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Goal](#goal)
- [Principle](#principle)
- [Naming Conventions](#naming-conventions)
  - [Repository](#repository)
  - [Executor classes](#executor-classes)
  - [File Names](#file-names)
- [Folder & Files](#folder--files)
  - [Fixed filenames and their Purposes](#fixed-filenames-and-their-purposes)
- [How to Write `manifest.yml`](#how-to-write-manifestyml)
  - [How to choose `keywords`?](#how-to-choose-keywords)
- [How to Write `setup.py`?](#how-to-write-setuppy)
- [How to Write `README.md`?](#how-to-write-readmemd)
- [How to write the docstrings?](#how-to-write-the-docstrings)
- [Do I need separated `Dockerfile` for CPUs and GPUs](#do-i-need-separated-dockerfile-for-cpus-and-gpus)
- [Development Checklist](#development-checklist)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Goal

We provide robust and easy-to-use executors for the Jina users. These executors are maintained officially by the Jina team under the github/jina-ai.  


# Principle

Keep one executor per repository. indexer is an exception. If you think it is necessary to have multiple executors in one repository, please raise this up in the `proj-hub` channel


# Naming Conventions


## Repository

The repositories should start with `jina-ai/executor-`. It is recommended to follow the convention below, 


```
jina-ai/executor-[modality_name]-[executor_name]
```



## Executor classes

No rules as long as you follow PEP8 and use the CamelCase.


## File Names

No rules as long as you follow PEP8 and use all lower cases with underscores if it helps improve the readability. 


# Folder & Files



*   On the presences of the files:
    *   💠 Must have, 
    *   🔸 Optional, 
    *   🔹 Free to have, e.g. user files
*   On the name of the files:
    *   ⭕ Must follow the file name Hubble/GithubAction defined
    *   🟢 Arbitrary file name
*   On the number of files:
    *   1️⃣ Only one
    *   🔢 Can be multiple


```
- jina-ai/executor-bar.git/  💠🟢1️⃣
    |- .github  💠⭕1️⃣
         |- workflow  💠⭕1️⃣
             |- ci.yml 💠⭕1️⃣
    |- .dockerignore  💠⭕1️⃣
    |- scripts/
        |- test.sh 💠⭕1️⃣
    |- setup.py  💠⭕1️⃣
    |- Dockerfile  💠⭕1️⃣
    |- README.md  💠⭕1️⃣
    |- requirements.txt  💠⭕1️⃣
    |- config.yml  💠⭕1️⃣
    |- manifest.yml  💠⭕1️⃣
    |- bar.py  💠🟢1️⃣
    |- helper(s).py  🔹🟢🔢
    |- hubble.yml  🔹⭕1️⃣
    |- tests/  💠⭕1️⃣
        |- __init__.py  💠⭕1️⃣
        |- requirements.txt  💠⭕1️⃣
	  |- unit/ 💠⭕1️⃣
        	|- test_bar.txt  💠🟢🔢
        |- integration/ 💠⭕1️⃣
      |- test_bar_integration  💠🟢🔢
```



## Fixed filenames and their Purposes


<table>
  <tr>
   <td>⭕<strong> Fixed file name</strong>
   </td>
   <td><strong>Purpose</strong>
   </td>
  </tr>
  <tr>
   <td><code>.github/</code>
   </td>
   <td>the Github Actions for running tests during CI
   </td>
  </tr>
  <tr>
   <td><code>.dockerignore</code>
   </td>
   <td>A list of files and folders that are not added to the docker image
   </td>
  </tr>
  <tr>
   <td><code>scripts/</code>
   </td>
   <td>the shell script needed for running tests during CI
   </td>
  </tr>
  <tr>
   <td><code>setup.py</code>
   </td>
   <td>The dynamic meta data used by <code>pip install</code> . It defines the python package and module information. It is needed by running the tests during CI
   </td>
  </tr>
  <tr>
   <td><code>Dockerfile</code>
   </td>
   <td>the Dockerfile that Hubble will build on
   </td>
  </tr>
  <tr>
   <td><code>README.md</code>
   </td>
   <td>the usage of the Executor
   </td>
  </tr>
  <tr>
   <td><code>requirements.txt</code>
   </td>
   <td><code>pip install -r</code> required packages
   </td>
  </tr>
  <tr>
   <td><code>config.yml</code>
   </td>
   <td>the Executor YAML config file
   </td>
  </tr>
  <tr>
   <td><code>manifest.yml</code>
   </td>
   <td>an annotation contains meta information of the Executor to get better appealing on Jina Hub
   </td>
  </tr>
  <tr>
   <td><code>bar.py</code>
   </td>
   <td>the place where the executor is actually defined.
   </td>
  </tr>
  <tr>
   <td><code>tests/</code>
   </td>
   <td>the place where unit tests and integration test are placed
   </td>
  </tr>
  <tr>
   <td><code>hubble.yml</code>
   </td>
   <td>the build config file for Hubble
   </td>
  </tr>
</table>



# How to Write `manifest.yml`



`manifest.yml` is optional.

`manifest.yml` annotates your image so that it can be managed by Hubble. To get better appealing on Jina Hub, you should carefully set `manifest.yml` to the correct values.


<table>
  <tr>
   <td><strong>Key</strong>
   </td>
   <td><strong>Description</strong>
   </td>
   <td><strong>Default</strong>
   </td>
  </tr>
  <tr>
   <td><code>manifest_version</code>
   </td>
   <td>The version of the manifest protocol
   </td>
   <td><code>1</code>
   </td>
  </tr>
  <tr>
   <td><code>name</code>
   </td>
   <td>Human-readable title of the image
   </td>
   <td>None
   </td>
  </tr>
  <tr>
   <td><code>alias</code>
   </td>
   <td>The Docker image name
   </td>
   <td>the digest of the Docker image
   </td>
  </tr>
  <tr>
   <td><code>description</code>
   </td>
   <td>Human-readable description of the software packaged in the image
   </td>
   <td>None
   </td>
  </tr>
  <tr>
   <td><code>author</code>
   </td>
   <td>Contact details of the people or organization responsible for the image (string)
<p>
By default, we use 
<p>
<code>Jina AI Dev-Team (dev-team@jina.ai)</code>
   </td>
   <td>None
   </td>
  </tr>
  <tr>
   <td><code>url</code>
   </td>
   <td>URL to find more information on the image (string)
   </td>
   <td>None
   </td>
  </tr>
  <tr>
   <td><code>avatar</code>
   </td>
   <td>A picture that personalizes and distinguishes your image
   </td>
   <td>None
   </td>
  </tr>
  <tr>
   <td><code>keywords</code>
   </td>
   <td>A list of strings help user to filter and locate your package
   </td>
   <td>None
   </td>
  </tr>
</table>



## How to choose `keywords`?


<table>
  <tr>
   <td>Category
   </td>
   <td>Possible <code>keywords</code>
   </td>
  </tr>
  <tr>
   <td>Data type
   </td>
   <td><code>image, text, audio, pdf</code>
   </td>
  </tr>
  <tr>
   <td>Modality
   </td>
   <td><code>multimodal, crossmodal</code>
   </td>
  </tr>
  <tr>
   <td>Framework
   </td>
   <td><code>PyTorch, TensorFlow, JAX, Transformers, Rust, Flair, TF Lite, Scikit-learn, ONNX, spaCy, allennlp, PaddlePaddle, NumPy</code>
   </td>
  </tr>
  <tr>
   <td>Language
   </td>
   <td><code>en, es, fr, de, zh, multilingual </code>
   </td>
  </tr>
  <tr>
   <td>Executor type
   </td>
   <td><code>encoder, indexer, ranker, evaluator, segmenter</code>
   </td>
  </tr>
  <tr>
   <td>Model name
   </td>
   <td>The model used in the executor. The name should be all in lowercase with underscores if necessary.
   </td>
  </tr>
</table>


Example `manifest.yml`:


```
manifest_version: 1 
name: CLIPImageEncoder 
alias: clip_image_encoder 
description: Encoder based on the Clip model proposed in https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf 
author: Jina AI Dev-Team (dev-team@jina.ai) 
url: https://jina.ai 
avatar: https://avatars.githubusercontent.com/u/60539444?s=200&v=4 
keywords: [clip, image, crossmodal, encoder, PyTorch, en]
```



# How to Write `setup.py`?

Important fields to customize


<table>
  <tr>
   <td>fields
   </td>
   <td>explanation
   </td>
  </tr>
  <tr>
   <td><code>url</code>
   </td>
   <td>The url to the repository of the executor
   </td>
  </tr>
  <tr>
   <td><code>py_modules</code>
   </td>
   <td>The package and module information. It should be `<code>jinahub.[indexer|encoder|segmenter|ranker|].module_name</code>` or `<code>jinahuab.module_name</code>`
   </td>
  </tr>
  <tr>
   <td><code>package_dir</code>
   </td>
   <td>This field is to map the package to the current folder. It should be <code>`jinahub.[indexer|encoder|segmenter|ranker|]`: `.`</code>
   </td>
  </tr>
</table>


Example `setup.py:`


```
import setuptools

setuptools.setup(
    name='jinahub-clip',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executors that encode images',
    url='https://github.com/jina-ai/executor-clip-image',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.clip_image'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
```



# How to Write `README.md`?

The README.md must include the following sections. Please refer to the [https://github.com/jina-ai/executor-clip-image](https://github.com/jina-ai/executor-clip-image) as a template.


<table>
  <tr>
   <td>Section
   </td>
   <td>Descriptions
   </td>
  </tr>
  <tr>
   <td>Introduction
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Prerequisites[Optional]
   </td>
   <td>specify the conditions before running the executors. For example, a pretrained model must be downloaded beforehand.
   </td>
  </tr>
  <tr>
   <td>Usages
   </td>
   <td>demonstrate how to add the executor into a Flow.
   </td>
  </tr>
  <tr>
   <td>Examples
   </td>
   <td>demonstrate how to use the executor in a Fow with codes showing the inputs and outputs
   </td>
  </tr>
  <tr>
   <td>Parameters
   </td>
   <td>specify the inputs and returns of the executor when processing requests
   </td>
  </tr>
  <tr>
   <td>Reference[Optional]
   </td>
   <td>
   </td>
  </tr>
</table>



# How to write the docstrings?

We follow the Jina Docstring Guideline at [https://docs1.jina.ai/chapters/docstring/](https://docs1.jina.ai/chapters/docstring/)

The ``__init__`` function and public functions of the executor must be documented. 


# Do I need separated `Dockerfile` for CPUs and GPUs

It is nice to have but not necessary. Hubble is only using `Dockerfile` at the moment. Please name the GPU one as `Dockerfile.gpu`.

# Development Checklist


- [ ] Is the repository following the naming convention?
- [ ] Is the folder structure well defined?
- [ ] Is the `manifest.yml` following the hubble convention? Are all the keywords legally defined?
- [ ] Is the `setup.py` well written?
- [ ] Is the `README.md` well written?
- [ ] Are the codes well documented with docstrings?
- [ ] Are there unit tests?
- [ ] Are there integration tests of running executors in a Flow?
Is there a `Dockerfile`? 