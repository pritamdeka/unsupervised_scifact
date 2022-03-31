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
  --top_n_abstracts top_3 \
  --top_n_sentences top_3 \
  --access_token for_HF_private_repo
```
The ```top_n_abstracts``` and ```top_n_sentences``` have default values of ```top_3``` respectively. The argument values that can be used for ```top_n_abstracts``` can be chosen from either of ```top_3```, ```top_5```, ```top_10``` which will extract the top 3, 5 and 10 abstracts for each claim respectively. The argument values that can be passed to ```top_n_sentences``` can be chosen from either ```top_1```, ```top_2```, ```top_3```, ```top_4```, ```top_5```, ```rms``` and ```am```. For the first 5 of those, it will extract the top 1, 2, 3, 4 and 5 sentences respectively which are similar to the claims. The ```rms``` and ```am``` will select those sentences which have a similarity higher than the ```root mean square``` or ```arithmetic mean``` of all similarities with the extracted sentences respectively.
