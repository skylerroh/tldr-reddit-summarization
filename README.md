# tldr-reddit-summarization
This project explores the use of hierarchical transformer-based model for summarization of forum posts across multiple subreddits on reddit.com.
The motivation behind the hierarchical model is to distil longer text down to only the most relevant top-k sentences before applying a encoder-decoder transformer 

## Abstract
Abstractive summarization has made substantial progress in recent years due to the introduction of transformers along with various self-supervised pretraining techiniques. While the majority of research in this area has been performed on news article datasets, other forms of text have not been widely tested. Other mediums such as posts on the web forum Reddit are significantly less structured. In this paper, a hierarchical model composed of a transformer-based sentence encoder and additional multihead-attention for sentence selection is proposed to better retain salient information while remaining within sequence length limits imposed by transformer architectures.
[Paper: Abststractive Summarization of Social Media Using Multiple Transformer Stages] (https://github.com/skylerroh/tldr-reddit-summarization/blob/master/Abstractive_Summarization_of_Social_Media_Using_Multiple_Transformer_Stages___Skyler_Roh.pdf).

## Dataset
[TL;DR: Mining Reddit to Learn Automatic Summarization, EMNLP 2017 - New Frontiers in Summarization workshop](https://github.com/webis-de/webis-tldr-17-corpus).   
`curl https://zenodo.org/record/1043504/files/corpus-webis-tldr-17.zip?download=1 --output corpus-webis-tldr-17.zip`
`unzip corpus-webis-tldr-17.zip`

## Environments
### Pegasus
`conda create --name pegasus_env python=3.7`
`conda activate pegasus_env`
`cd pegasus && pip install -r requirements.txt`

## Notebooks and Training Scripts
### Data Processing and EDA
`eda/DataProcessing.ipynb`  
  - read in data in original json form
  - determine frequency of various subreddits in dataset
  - inspect proportion of unknown token occurences   
  
`eda/Baseline_ROUGE.ipynb`  
  - examine baseline rouge scores of basic extraction from the text: first 128 tokens, last 128 tokens, first 64 tokens + last 64 tokens   
  
`preprocess/TFRecords.ipynb`  
  - filter records with titles and subreddits that are non-null
  - write each post into TFRecord format
  - inputs = title + subreddit + post text.  
  
### Models 
**Extractive**
`ExtractiveSentenceSelection_HiBert.ipynb`  
  - ingest TFRecords and process into input form encoding each document as list[sentences] where each sentence is list[wordpiece tokens]
  - create model architecture that 
    - encodes each sentence via pretrained distilbert
    - pass encoded sentences into layer of multiheadattention
    - feed-forward dense layers
    - predict probability of salience
    - rank top-k sentences and concat chronologically into a new text input for PEGASUS encoder-decoder for abstractive summarization
   
**Abstractive**
PEGASUS  
Run on TF 1.15 on gcp on n1-highmem-8 (8cpu, 52gb ram) with an nvidia tesla T4, batch sizes and number of epochs able to run limited by compute capacity and time.

Training example: `python3 pegasus/bin/train.py --params=reddit_tldr_subreddit_samples_extracted20 --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,learning_rate=0.0001,batch_size=4,max_input_len=512,train_steps=50000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/reddit_tldr_w_title_extracted20`

Eval example: `python3 pegasus/bin/evaluate.py --params=reddit_tldr_subreddit_samples_extracted20 --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=4,max_input_len=512,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/reddit_tldr_w_title_extracted20/model.ckpt-50000`
    
