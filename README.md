# Identifying Nuances in Fake News vs. Satire: Using Semantic and Linguistic Cues (NLP4IF, EMNLP-IJCNLP 2019)

[![DOI:10.18653/v1/D19-5004](https://zenodo.org/badge/DOI/10.18653/v1/D19-5004.svg)](https://doi.org/10.18653/v1/D19-5004)

#### Description of classes:
* **classify_satire_fake.py**: this code implements a Multinational Naive Bayes text classifier, as described in the paper of Golbeck et al. 2018 (Fake news vs satire: A dataset and analysis).

#### Files in data folder:
In all of the following file, `0` and `1` are the labels for `fake` and `satire` articles, respectively.

* **data/satire_fake_full.xlsx**: this is the file including all the indexes from Coh-Metrix. This file is our input in all of our experiments in R.
* **data/classification.csv**: this file includes all the significant components from our regression analysis in R. We use this file as our input for the binary classification task.

---
### Citation Information
If you found our work or any insight we report interesting, please use the following information to cite our paper:

```
@inproceedings{levi-etal-2019-identifying,
    title = "Identifying Nuances in Fake News vs. Satire: Using Semantic and Linguistic Cues",
    author = "Levi, Or and Hosseini, Pedram and Diab, Mona and Broniatowski, David",
    booktitle = "Proceedings of the Second Workshop on Natural Language Processing for Internet Freedom: Censorship, Disinformation, and Propaganda",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5004",
    pages = "31--35",
}
```
