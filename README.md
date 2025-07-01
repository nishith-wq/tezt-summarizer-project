# tezt-summarizer-project

*COMPANY* - *CODTECH IT SOLUTIONS*

*NAME* - *NISHEETH CHANDRA*

*INTERN ID* - *CT04DF2787*

*DOMAIN* - *ARTIFICIAL INTELLIGENCE*

*DURATION* - *4 WEEKS*

*MENTOR* - *NEELA SANTHOSH*

# Text Summarization Using NLP and TextRank

## Overview

This project implements an unsupervised extractive text summarization system using Natural Language Processing (NLP) techniques in Python. It showcases how raw textual data can be parsed, processed, and distilled into concise summaries using an algorithmic approach. The core model draws inspiration from the TextRank algorithm, which is based on Google’s PageRank concept, adapted for natural language processing. It uses word embeddings and graph-based ranking to identify the most significant sentences in a document.

The primary objective of this notebook is to demonstrate how a well-structured NLP pipeline can be used to summarize complex textual data while preserving semantic integrity. The process utilizes Python libraries such as SpaCy, string processing, and NetworkX for building the graph model.

## Problem Statement

In the digital age, we are constantly bombarded with large volumes of textual data—be it news articles, research papers, or social media posts. Human attention, however, is a limited resource. There arises a need for systems that can automatically generate accurate and coherent summaries of large texts to facilitate quick understanding and decision-making.

## Objectives

- Tokenize and preprocess input text
- Eliminate irrelevant tokens such as punctuation and stop words
- Calculate word frequencies for significance scoring
- Score sentences based on cumulative word importance
- Use graph-based ranking (TextRank) to select the most informative sentences
- Generate a coherent summary as output

## Methodology

### 1. **Text Preprocessing**
The input text is first cleaned by removing newline characters, punctuation, and common stop words (like "the", "and", "is") which do not contribute meaningfully to the summarization task. This step uses SpaCy’s tokenization features and Python’s built-in `string` module.

### 2. **Word Frequency Calculation**
Each word in the cleaned text is analyzed to calculate its frequency of occurrence. A dictionary is created to hold the word counts, which are later normalized to ensure fair comparison. These frequencies are used to assign weigh

