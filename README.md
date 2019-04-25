# Code for the paper "Unsupervised Community Detection with Modularity-based Attention Model"

Installation
------------
```bash
$ git clone https://github.com/Ivanopolo/modnet
$ python3.6 -m venv modnet_env
$ source modnet_env/bin/activate
$ cd modnet
$ pip install -r requirements.txt
```

How to reproduce results
------------
```bash
$ python generate_datasets.py
$ python evaluate_models.py graphs/associative_n=400_k=5/test
$ python evaluate_models.py --disassociative graphs/disassociative_n=400_k=5/test
```
