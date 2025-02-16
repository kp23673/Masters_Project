# Masters-Project
Predicting Cancer Type from Symptom Descriptions

## About Dataset

- For Biomedical text document classification, abstract and full papers(whose length less than or equal to 6 pages) available and used. This dataset focused on long research paper whose page size more than 6 pages. Dataset includes cancer documents to be classified into 3 categories like 'Thyroid_Cancer','Colon_Cancer','Lung_Cancer'.
  
- Total publications=7569. it has 3 class labels in dataset.
number of samples in each categories:
colon cancer=2579, lung cancer=2180, thyroid cancer=2810

## Problem Statement:

The early detection of cancer is crucial for effective treatment, but many patients initially present with vague or ambiguous symptoms. Accurately predicting the type of cancer based on symptom descriptions can help in early diagnosis and appropriate referral. This project aims to develop a machine learning model that can predict the type of cancer from a patient's symptom description using natural language processing (NLP) techniques, providing a potential tool for healthcare professionals to support early detection and improve patient outcomes.

## Description:
- Built a pipeline to predict cancer types from textual symptom descriptions using Python and NLTK.
- Performed data preprocessing, analysis, and visualization to explore patterns in symptoms.
- Trained machine learning models like Random Forest, SVM, Logistic Regression, Decision Tree, and Naive Bayes.
- Implemented LSTM and BERT models to improve prediction accuracy using deep learning.
- Evaluated models using accuracy, precision, recall, and F1-score to ensure optimal performance.

## Insights
- Advanced deep learning models, such as LSTM and Transformer-based BERT, significantly outperform traditional machine learning models in cancer type classification&#8203;:contentReference[oaicite:0]{index=0}.
- The use of pipelines in text classification improves workflow efficiency by automating text vectorization and classification&#8203;:contentReference[oaicite:1]{index=1}.
- Evaluation metrics like accuracy, precision, recall, and F1-score confirm the reliability of machine learning models, with BERT achieving an F1-score of 95.11%&#8203;:contentReference[oaicite:2]{index=2}.
- Future advancements in ensemble approaches and multi-label classification can further enhance predictive accuracy and practical applications&#8203;:contentReference[oaicite:3]{index=3}.

## Recommendations
- Implement ensemble learning techniques to integrate multiple models and enhance predictive performance.
- Develop an intuitive AI-powered tool for medical professionals to input symptom descriptions and receive diagnostic insights.
- Focus on ethical considerations and model interpretability to ensure AI predictions are transparent and trusted by stakeholders&#8203;:contentReference[oaicite:4]{index=4}.
- Leverage cutting-edge deep learning architectures like GPT or T5 for enhanced contextual understanding in medical text classification.

## Skills Learned & Usefulness for Stakeholders
### Skills Acquired:
- **Machine Learning & Deep Learning**: Gained expertise in Logistic Regression, Decision Trees, Random Forests, SVM, Naive Bayes, LSTM, and BERT models.
- **Natural Language Processing (NLP)**: Learned text preprocessing techniques such as tokenization, CountVectorizer, TF-IDF, and Named Entity Recognition (NER).
- **Model Evaluation & Optimization**: Mastered confusion matrices, classification reports, hyperparameter tuning, and performance enhancement strategies.

### Usefulness for Stakeholders:
- **Healthcare Professionals**: AI-powered classification tools can aid in faster and more accurate cancer diagnosis.
- **Medical Researchers**: Insights from AI-driven classification can assist in understanding disease patterns and improving diagnostic algorithms.
- **AI & Data Scientists**: Enhancing medical NLP applications through innovative AI techniques.
- **Patients & Caregivers**: Early and accurate classification of cancer types can lead to timely treatment decisions&#8203;:contentReference[oaicite:5]{index=5}.
