# Representation learning: propositionalization and embeddings

This repository is a growing collection of supplementary material for our book [Representation learning: propositionalization and embeddings](). Currently, a number of Jupyter notebooks illustrating the selected parts of the book are available.

## Table of contents

1. Introduction to Representation Learning
2. Machine Learning Background
3. Text Embeddings
    1. [LSA and LDA](../master/Chapter3/LSA_LDA.ipynb)
    2. [word2vec](../master/Chapter3/word2vec.ipynb)
    3. [BERT](../master/Chapter3/BERT.ipynb)
4. Propositionalization of Multi-Relational Data
    1. [Wordification](../master/Chapter4/wordification.ipynb)
    2. [python-rdm](../master/Chapter4/python-rdm.ipynb)
5. Graph and Heterogeneous Network Transformations
    1. [node2vec](../master/Chapter5/node2vec.ipynb)
    2. [metapath2vec](../master/Chapter5/metapath2vec.ipynb)
    3. [hinmine](../master/Chapter5/hinmine.ipynb)
6. Unifying approaches
    1. [StarSpace](../master/Chapter6/starspace.ipynb)
    2. [propDRM](../master/Chapter6/propDRM.ipynb)


## How to use

### Requirements

- python 3.8 (3.6 and 3.7 were also tested and should work as well)
- jupyterlab

In addition, each notebook has its own requirements which are installed when the notebook is executed for the first time.

### Preparing the environment

1. Create and activate a virtual environment.

    - Linux
      ```bash
      python3 -m venv myEnv
      source myEnv/bin/activate
      ```
  
    - Windows
      ```bash
      python3 -m venv myEnv
      myEnv\Scripts\activate
      ```
      
2. Install and run jupyterlab. The following commands install jupyterlab and returns a link to the web application.
    ```bash
    pip install jupyterlab
    jupyter lab
    ```
3. Open the link in a web browser.

4. Upload the selected notebook and run it.


## How to contribute

Contributions are welcome! You are welcome to contribute corrections, new notebooks, examples, figures or any other material related to the contents of the book.

## License

The code and materials in this repository are licensed under the MIT license except where stated otherwise.
