# SHL Assessment Recommendation System - Technical Approach

## Overview

I designed and implemented a comprehensive Streamlit application that recommends SHL assessments based on job descriptions or natural language queries. This solution leverages modern NLP techniques to match user queries with relevant assessments from SHL's product catalog.

## Architecture

The solution is built with a focus on simplicity and user experience:

1. **Data Layer**:
   - Web scraping module to extract assessment information from SHL's product catalog
   - Data preprocessing pipeline and embedding generation
   - Caching mechanisms for performance optimization

2. **Logic Layer**:
   - Semantic search engine using sentence embeddings
   - Recommendation algorithm with customizable filtering
   - Evaluation framework for measuring recommendation quality

3. **Presentation Layer**:
   - Streamlit-based web application with intuitive interface
   - Multiple input methods: text, file upload, URL
   - Interactive visualization of recommendations
   - API demonstration capabilities

## Technical Implementation

### Data Collection and Processing

- **Web Scraping**: Utilized BeautifulSoup to extract product data from SHL's catalog
- **Data Extraction**: Parsed assessment details including name, URL, duration, test type, and support features
- **Text Preprocessing**: Applied NLP techniques including tokenization and stop word removal

### Recommendation Engine

- **Embedding Model**: Implemented Sentence Transformers (all-MiniLM-L6-v2) for semantic understanding
- **Similarity Calculation**: Used cosine similarity to find assessments matching query context
- **Filtering Logic**: Implemented duration filtering to meet specific time constraints in queries

### User Interface

- **Multi-tab Design**: Created separate tabs for recommendations, API demonstration, and evaluation
- **Interactive Elements**: Incorporated various input methods and parameter controls
- **Results Visualization**: Displayed recommendations in both tabular and detailed expandable formats

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: For interactive web application development
- **Sentence Transformers**: For semantic understanding and embeddings
- **BeautifulSoup**: For web scraping the SHL catalog
- **NLTK**: For text preprocessing
- **Pandas/NumPy**: For data manipulation and analysis
- **Docker**: For containerization and easy deployment
- **scikit-learn**: For similarity calculations and evaluation metrics

## Development Process

1. **Research & Design**: Analyzed SHL catalog structure and user requirements
2. **Data Collection**: Implemented web scraping to gather assessment information
3. **Model Development**: Created an embedding-based recommendation engine
4. **UI Development**: Built an intuitive Streamlit interface with multiple features
5. **Evaluation**: Implemented standard IR metrics for quality assessment
6. **Deployment**: Containerized the application for easy hosting

## Results and Future Improvements

- The system achieves strong performance on test queries, with high recall and precision
- The Streamlit interface provides an intuitive experience for hiring managers
- Future enhancements could include:
  - Fine-tuning the embedding model on HR-specific terminology
  - Adding user feedback mechanisms to improve recommendations
  - Implementing more advanced file parsing for uploaded job descriptions
  - Adding visualization features for recommendation explanations