# Code for the paper ""
The operation allows to use approximate nearest neighbors search to do faster top-k retrievals over a set of embeddings with the assumption that the underlying embeddings do not change very quickly.

Installation
------------
```bash
$ git clone https://github.com/Ivanopolo/modnet
$ python3.6 -m venv modnet_env
$ source modnet_env/bin/activate
$ cd modnet
$ pip install -r requirements.txt
```

How to reproduce paper results
------------

```bash
$ python generate_datasets.py
$ python evaluate_models.py graphs/associative_n=400_k=5/test
$ python evaluate_models.py --disassociative graphs/disassociative_n=400_k=5/test
```
