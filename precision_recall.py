from beir import util, LoggingHandler

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import numpy as np
import ast
import math
import collections
import logging
import pathlib, os
import argparse
from tqdm import tqdm
import time
import statistics
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
from beir.retrieval import models
import random
from transformers import TextClassificationPipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import jsonlines
import pandas as pd


if __name__ == "__main__":

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--rct_model_name", type=str)
  argparser.add_argument("--sbert_model_name", type=str)
  argparser.add_argument('--dev_file', type=str)
  argparser.add_argument('--corpus_file', type=str)
  argparser.add_argument("--access_token", type=str)
  argparser.add_argument("--top_n_sentences", type=str, default="top_3") 
  argparser.add_argument("--top_n_abstracts", type=str, default="top_3")   

  args, unknown = argparser.parse_known_args()

  rct_model_name = args.rct_model_name              ######### pubmed200krct model
  sbert_model_name = args.sbert_model_name          ######### sbert model  
  dev_file = args.dev_file
  corpus_file = args.corpus_file            
  access_token = args.access_token
  top_n_sentences = args.top_n_sentences
  top_n_abstracts = args.top_n_abstracts
  
    
  def rmsValue(num_list,n):
    square=0
    mean=0.0
    root=0.0
    for i in range(0,n):
      square += (num_list[i]**2)
    mean = (square/ (float)(n))
    root = math.sqrt(mean)
    return root

  model_rct = AutoModelForSequenceClassification.from_pretrained(rct_model_name, use_auth_token=access_token)
  tokenizer = AutoTokenizer.from_pretrained(rct_model_name, use_auth_token=access_token)
  pipe = TextClassificationPipeline(model=model_rct, tokenizer=tokenizer, return_all_scores=True, device=0, batch_size=512)
  model_sbert = SentenceTransformer(sbert_model_name)

  #### Just some code to print debug information to stdout
  logging.basicConfig(format='%(asctime)s - %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S',
                      level=logging.INFO,
                      handlers=[LoggingHandler()])
  #### /print debug information to stdout  

  all_sent=[]
  all_claims=[]
  corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus_file)}
  for claim in jsonlines.open(dev_file):
    all_claims.append(claim['claim'])
    sent_list=[]
    for doc_id in claim["cited_doc_ids"]:    
      doc = corpus[int(doc_id)]
      if "discourse" in doc:
        abstract_sentences = \
        [discourse + " " + sentence for discourse, sentence in zip(doc['discourse'], doc['abstract'])]
      else:
        abstract_sentences = doc['abstract']    
      doc_id = str(doc_id)    
      if doc_id in claim['evidence']:      
        evidence = claim['evidence'][doc_id]
        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
        sentence_list=[abstract_sentences[edx] for edx in evidence_sentence_idx]
        sent_list.append(sentence_list)
    all_sent.append(sent_list)  
  df=pd.DataFrame((zip(all_claims, all_sent)),
                columns=['claims','sentences'])

  with jsonlines.open(dev_file, 'r') as jsonl_f:
    lst = [obj for obj in jsonl_f]
    id_list = [l['id'] for l in lst]
    claim_list= [l['claim'] for l in lst]
    mydict=dict(zip(id_list,claim_list))
    df['id'] = id_list
  df=df[df['sentences'].map(lambda d: len(d)) > 0]
  df['sentences'] = df.sentences.apply(lambda x: sum(x, []))
  gold_sentence_list=list(df['sentences'])


  dataset = "scifact"
  url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
  out_dir = os.path.join(os.getcwd(), "datasets")
  data_path = util.download_and_unzip(url, out_dir)
  print("Dataset downloaded here: {}".format(data_path))
  
  data_path = "datasets/scifact"
  corpus, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"
  dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
  z=list(df['id'])
  z=[str(x) for x in z]
  small_dict=dict_filter(queries, z)
  model_abstract = DRES(models.SentenceBERT("pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT"), batch_size=128)
  retriever = EvaluateRetrieval(model_abstract, score_function="cos_sim")
  results = retriever.retrieve(corpus, small_dict)

  qids_list=(list(results.keys()))
  docid_list=(list(results.values()))
  docid_list_sorted=[dict(sorted(al.items(), key=lambda x: x[1], reverse=True)) for al in docid_list]
  overall_docid_list=[]
  for z in docid_list_sorted:
    if(top_n_abstracts == "top_3"):
      top_abst = 3
    elif(top_n_abstracts == "top_5"):
      top_abst = 5
    elif(top_n_abstracts == "top_10"):
      top_abst = 10
    temp_list=list(z.keys())
    temp_list=temp_list[:top_abst]
    overall_docid_list.append(temp_list)
  qrels_dict=dict_filter(qrels, qids_list)
  qrels_values_list=list(qrels_dict.values())
  qrels_id_list=[list(j.keys()) for j in qrels_values_list]
  retrieved_id_list=[[ast.literal_eval(i) for i in k] for k in overall_docid_list]
  k_list=[[ast.literal_eval(i) for i in k] for k in qrels_id_list]


  corpus_1 = {doc['doc_id']: doc for doc in jsonlines.open(corpus_file)}
  y_list=retrieved_id_list
  final_sentence_list=[]
  for el in tqdm(y_list, desc = 'Sentence Extraction'): 
    abs_list=[corpus_1[e]['abstract'] for e in el]
    all_sentence_list=[]
    for elements in abs_list:
      z=pipe(elements)
      l_list=[list(max(el, key=lambda x:x['score']).values())[0] for el in z]
      mydict = dict(zip(elements,l_list))
      sentence_list=[]
      for k,v in mydict.items():
        if(v == 'CONCLUSIONS' or v == 'RESULTS'):     
          sentence_list.append(k)
      all_sentence_list.append(sentence_list)
    flat_list = [item for sublist in all_sentence_list for item in sublist]
    final_sentence_list.append(flat_list)
  print("The final list of result/conclusion sentences:", final_sentence_list)
  
  claim_list = list(small_dict.values())
  final_ranked_sentence_list=[]
  for i in range(len(qids_list)):
    candidates=final_sentence_list[i]
    doc_embedding = model_sbert.encode([claim_list[i]])
    if candidates==[]:
      keywords = []
      final_ranked_sentence_list.append(keywords)
    else:       
      candidate_embeddings = model_sbert.encode(candidates)
      distances = cosine_similarity(doc_embedding, candidate_embeddings)
      arr = np.asarray(distances)
      if(top_n_sentences == "top_1"):
        top_n=1
      elif(top_n_sentences == "top_2"):
        top_n=2
      elif(top_n_sentences == "top_3"):
        top_n=3
      elif(top_n_sentences == "top_4"):
        top_n=4
      elif(top_n_sentences == "top_5"):
        top_n=5
      elif(top_n_sentences == "rms"): 
        num_list=[z for i in arr for z in i if z>0 ]
        n=len(num_list)
        z=(rmsValue(num_list,n))
        top_n=((arr > z).sum())
      elif(top_n_sentences == "am"):
        num_list=[z for i in arr for z in i if z>0 ]
        n=len(num_list)
        z=statistics.mean(num_list)
        top_n=((arr > z).sum())
      keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]] 
      final_ranked_sentence_list.append(keywords)
  print("The final ranked sentences list:", final_ranked_sentence_list)
  print("Gold sentence list:", gold_sentence_list)
  precision_list=[]
  recall_list=[]
  for elem,elem1 in zip(final_ranked_sentence_list,gold_sentence_list):
    len_elem=(len(elem))
    len_all=len(elem1)
    list3 = set(elem)&set(elem1)
    list4 = sorted(list3, key = lambda k : elem.index(k)) #sort the above
    temp=(len(list4))
    precision=float(temp/len_elem)
    precision_list.append(precision)
    if len_elem == 0:
      recall=0
      recall_list.append(recall)
    else:
      recall=float(temp/len_all)
      recall_list.append(recall)  
  mean_precision=(statistics.mean(precision_list))*100
  mean_recall=(statistics.mean(recall_list))*100
  f_score = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
  print("Precision %:", mean_precision)
  print("Recall %:", mean_recall)
  print("F-1 score %:", f_score)
