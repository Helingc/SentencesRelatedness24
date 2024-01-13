# Semantic Textual Relatedness Analysis (STR2022)
## A Project in Applied Deep Learning
#### University of Munich, Winter Semester 2023/2024

#### Authors:
- **Daniel Chrościcki**
- **Lennart Marx**

---

## Overview
This project is inspired by the research article ["What Makes Sentences Semantically Related? A Textual Relatedness Dataset and Empirical Study"](https://aclanthology.org/2023.eacl-main.55.pdf). It focuses on exploring and expanding upon the findings of the article, specifically the performance of different Natural Language Processing (NLP) models on the STR2022 dataset which is presented within the original paper and includes 5,500 manually annotated English sentence pairs. 

### Experimentation and Methodology
In our journey to understand and expand upon the original findings, we have:
- **Reproduced experiments** from the original paper, assessing the robustness and reproducibility of the reported results.
- **Explored new methods** and techniques, including various model architectures and pre-trained models, to enhance our understanding and application of semantic relatedness.
- **Experimented with a diverse range of approaches and classes of models**, including:
  - **Lexical Overlap**: Assessing relatedness based on the overlap of words and phrases.
  - **Static Embeddings**: Utilizing fixed word embeddings for semantic analysis.
  - **Contextual Embeddings**: Implementing models that use embeddings sensitive to the context of words in a sentence.
  - **Tuned Model**: Customizing and fine-tuning existing models for better performance on our dataset.
  - **Tuned Large Language Models (LLM)**: Adapting and tuning large-scale language models to our specific use case.
  - **LLM Chat**: Exploring the use of conversational AI models to analyze and predict semantic relatedness.

Our comprehensive approach not only validates previous findings but also opens new avenues for understanding semantic textual relatedness through a variety of NLP methodologies.


## Repository Structure
- **data/**: Contains the training and testing datasets
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
2. Choose and open a notebook in Jupyter Notebook. Code is structured to use GPU acceleration when available. It's also suitable for running using Google Colab Notebooks.
3. Execute the notebook cells to train and evaluate the selected model.


## Acknowledgments

- We extend our special thanks to **Ercong Nie**, our project supervisor from the Center for Information and Language Processing (CIS) at Ludwig Maximilians University of Munich (LMU Munich).
- Heartfelt gratitude to **Professor David Rügamer**, Associate Professor at LMU Munich and head of the Data Science Group, for his invaluable guidance in the Applied Deep Learning course.
- Appreciation is also due to the authors of the original article that inspired this project, contributing significantly to our understanding and exploration of semantic textual relatedness.


## Project research document 
[Google doc](https://docs.google.com/document/d/1IxB6a3DGFe2ermGPOAnsYhukMOtDKvWfV2UXoyiqcYQ/edit#heading=h.bw9p4docrkd1)
