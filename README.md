# tldr-reddit-summarization
This project explores the use of hierarchical transformer-based model for summarization of forum posts across multiple subreddits on reddit.com.
The motivation behind the hierarchical model is to distil longer text down to only the most relevant top-k sentences before applying a encoder-decoder transformer 

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
`ExtractiveSentenceSelection_HiBert.ipynb`  
  - ingest TFRecords and process into input form encoding each document as list[sentences] where each sentence is list[wordpiece tokens]
  - create model architecture that 
    - encodes each sentence via pretrained distilbert
    - pass encoded sentences into layer of multiheadattention
    - feed-forward dense layers
    - predict probability of salience
    - rank top-k sentences and concat chronologically into a new text input for PEGASUS encoder-decoder for abstractive summarization
    
