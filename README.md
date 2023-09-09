# HeteGCN: Heterogeneous Graph Convolutional Networks for Text Classification

This repository contains the source code for our WSDM (Web Search and Data Mining) 2021 [paper](https://dl.acm.org/doi/10.1145/3437963.3441746).

## Citation

Please consider citing the following paper when using our code.

```bibtex
@inproceedings{10.1145/3437963.3441746,
author = {Ragesh, Rahul and Sellamanickam, Sundararajan and Iyer, Arun and Bairi, Ramakrishna and Lingam, Vijay},
title = {HeteGCN: Heterogeneous Graph Convolutional Networks for Text Classification},
year = {2021},
isbn = {9781450382977},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3437963.3441746},
doi = {10.1145/3437963.3441746},
booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
location = {Virtual Event, Israel},
series = {WSDM '21}
}
```


## Abstract
We consider the problem of learning efficient and inductive graph convolutional networks for text classification with a large number of examples and features. Existing state-of-the-art graph embedding based methods such as predictive text embedding (PTE) and TextGCN have shortcomings in terms of predictive performance, scalability and inductive capability. To address these limitations, we propose a heterogeneous graph convolutional network (HeteGCN) modeling approach that unites the best aspects of PTE and TextGCN together. The main idea is to learn feature embeddings and derive document embeddings using a HeteGCN architecture with different graphs used across layers. We simplify TextGCN by dissecting into several HeteGCN models which (a) helps to study the usefulness of individual models and (b) offers flexibility in fusing learned embeddings from different models. In effect, the number of model parameters is reduced significantly, enabling faster training and improving performance in small labeled training set scenario. Our detailed experimental studies demonstrate the efficacy of the proposed approach.

## Reproducing Results
To reproduce the results presented in the paper, you can utilize the bash scripts available in the `bash_scripts` directory. Please refer the following notations used in this repository to associate the results in the paper.

| Code | Paper |
| --- | --- |
| FF-NF | HeteGCN(F-X) |
| NF-FN-NF | HeteGCN(X-TX-X) |
| FN-NF | HeteGCN(TX-X) |

This code supports constructing arbitraty HeteGCN paths consisting of layers [FF, NF, FN]. The layers should be compatible. 

Example HeteGCN Paths
```
1. FF-NF
2. NF-FN-FN
3. FN-NF
```
 

## Steps To Run
1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Download the public datasets present in an drive link:
```python
python download_datasets.py
```

3. Example command to run
```python
python train.py  --data_dir datasets --dataset mr --path NF-FN-NF --wt_reg 10.0 --learning_rate 0.002 --dropout 0.25 --emb_reg 0.0 --FN_norm None --NF_norm sym --verbose
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.