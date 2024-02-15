# Semantic Textual Relatedness Analysis (STR2022)
## A Project in Applied Deep Learning
#### University of Munich, Winter Semester 2023/2024

#### Authors:
- **Daniel Chrościcki**
- **Lennart Marx**


## Overview
This project is inspired by the research article ["What Makes Sentences Semantically Related? A Textual Relatedness Dataset and Empirical Study"](https://aclanthology.org/2023.eacl-main.55.pdf). It focuses on exploring and expanding upon the findings of the article, specifically the performance of different Natural Language Processing (NLP) models on the STR2022 dataset which is presented within the original paper and includes 5,500 manually annotated English sentence pairs. 

The first part of our project focuses on reproducing the experiments from the original paper, most of which resulted in comparable performance. However, there were instances where our findings diverged. The second part of our work is dedicated to extending the research with new models, model classes and data augmentation methods. 

Notably, our project achieved the best cross-validation Spearman correlation score of 0.87, surpassing the 0.83 score reported in the original article. 

### Experimentation and Methodology
In our journey to understand and expand upon the original findings, we have:
- **Reproduced experiments** from the original paper, assessing the robustness and reproducibility of the reported results.
- **Explored new methods** and techniques, including various model architectures and pre-trained models, to enhance our understanding and application of semantic relatedness.
- **Experimented with a diverse range of approaches and classes of models**, including:
  - **Lexical Overlap**: Assessing relatedness based on the overlap of words and phrases.
    - [Dice Overlap](/notebooks/lexical_overlap_dice.ipynb)
  - **Static Embeddings**: Utilizing fixed word embeddings for semantic analysis.
    - [Fasttext](/notebooks/embeddings_static_fasttext.ipynb)
    - [Word2Vec](/notebooks/embeddings_static_word2vec.ipynb)
    - [Sent2Vec](/notebooks/embeddings_static_sent2vec.ipynb)
  - **Contextual Embeddings**: Implementing models that use embeddings sensitive to the context of words in a sentence.
    - [BERT](/notebooks/embeddings_contextual_bert.ipynb)
    - [BART](/notebooks/embeddings_contextual_bart.ipynb)
  - **Fine-tuned Models**: Customizing and fine-tuning existing models for better performance on our dataset.
    - [BERT](/notebooks/fine_tuned_model_bert.ipynb)
    - [BART](/notebooks/fine_tuned_model_bart.ipynb)
    - [sBERT](/notebooks/fine_tuned_model_sbert.ipynb)
    - [RoBERTa](/notebooks/fine_tuned_model_RoBERTa_augumentation.ipynb)
    - [T5](/notebooks/fine_tuned_model_t5.ipynb)
  - **Fine-tuned Large Language Models (LLM)**: Adapting and tuning large-scale language models to our specific use case.
    - [LLAMA2](/notebooks/fine_tuned_llm_llama2.ipynb)
  - **LLM Chat**: Exploring the use of conversational AI models to analyze and predict semantic relatedness.
    - [LLAMA2 Chat](/notebooks/llm_chat_llama2.ipynb)
- **Introducing data augmentation techniques**
    - **Data Augmentation**: Enhanced dataset diversity by introducing synonyms and letter replacements within sentences.
      - [RoBERTa](/notebooks/fine_tuned_model_RoBERTa_augumentation.ipynb)

### Results

- Replicated experiments generally aligned with the original studies, although occasional discrepancies emerged, especially when the specific version of the pre-trained model was unspecified.
- Contextual Embeddings demonstrated robust performance with swift inference speeds, requiring no fine-tuning.
- Fine-tuned NLP models, particularly BERT and BART-class, showcased superior performance, demanding less than 1 hour of computing time on GPU-backed Colab notebooks for training. This class of models allowed us to achieve a performance of 0.87 Spearman correlation score, compared to 0.83 from the original article. 
- LLM Models presented challenges during training, with smaller versions underperforming and larger ones being hard to run with the available resources. This class of models requires also an extensive tuning time.
- LLM models in chat mode encountered difficulties in accurately understanding the task, leading to unstable results despite using additional packages for controlling model behaviour.
- Introduced data augmentation did not yield significant performance improvements. However, we find this method might be worth further investigation and development.

Detailed experiment results are available in a tabular form in the results.csv file.  

## Repository Structure
- **data/**: Contains the training and testing datasets, as well as predictions that were submitted to the competition
- **notebooks/**: Contains Jupyter notebooks for training and evaluating NLP models.
- **results.csv**: Stores and presents the results of all trained models.


## Getting Started
### Prerequisites
- Python 3.10
- Jupyter Notebook

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/Helingc/SentencesRelatedness24
   ```
2. Install the necessary packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Navigate to the notebooks/ folder.
2. Choose and open a notebook in Jupyter Notebook. The code is structured to use GPU acceleration when available. It's also suitable for running using Google Colab Notebooks.
3. Execute the notebook cells to train and evaluate the selected model.


## Acknowledgments

- We express our gratitude to Ercong Nie, our project supervisor from the Center for Information and Language Processing (CIS) at Ludwig Maximilians University of Munich (LMU Munich).
- Thanks to Professor David Rügamer, Associate Professor at LMU Munich and head of the Data Science Group, for his guidance in the Applied Deep Learning course.
- We acknowledge the authors of the original article that inspired this project, whose work significantly contributed to our understanding and exploration of semantic textual relatedness.


## Project research document 
[Google Doc](https://docs.google.com/document/d/1IxB6a3DGFe2ermGPOAnsYhukMOtDKvWfV2UXoyiqcYQ/edit#heading=h.bw9p4docrkd1)
