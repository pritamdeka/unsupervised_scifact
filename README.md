# unsupervised_scifact

This repository contains the code to calculate precision, recall and f-score for SCIFACT dev set claims. The code can be run in Colab by using the instructions provided below. 

To run the code first clone the repository by the following command
```
!git clone https://github.com/pritamdeka/unsupervised_scifact.git
```

After that install the necessary requirements using the following.
```
!pip install -r requirements.txt
```
After that run the [precision_recall.py](https://github.com/pritamdeka/unsupervised_scifact/blob/main/precision_recall.py) file by using the following:

```
!python precision_recall.py \
  --rct_model_name pritamdeka/BioBert-PubMed200kRCT \
  --sbert_model_name any_sentence-transformers_model_from_Huggingface \
  --dev_file /path_to/claims_dev.jsonl \
  --corpus_file /path_to/corpus.jsonl \
  --access_token for_HF_private_repo
```
