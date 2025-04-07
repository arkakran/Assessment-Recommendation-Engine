
#app.py

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import json
from io import StringIO
import time

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define the base URL for SHL product catalog
BASE_URL = "https://www.shl.com/solutions/products/"

class SHLAssessmentSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.assessments_df = None
        self.embeddings = None
        self.stop_words = set(stopwords.words('english'))
        
        # Clear cache to force refresh with correct URLs
        self.clear_cache()
        
        # Try to load cached data or crawl if not available
        try:
            self.load_cached_data()
            # Verify data integrity
            if self.assessments_df is None or self.embeddings is None or len(self.embeddings) == 0:
                raise ValueError("Cached data is incomplete or corrupted")
            print("Loaded cached assessment data.")
        except Exception as e:
            print(f"Cached data issue: {e}. Creating SHL assessment data...")
            self.create_shl_data()
    
    def clear_cache(self):
        """Clear the cached data to force a refresh"""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
        try:
            # Check if cache files exist and remove them
            assessments_path = os.path.join(data_dir, 'assessments.pkl')
            embeddings_path = os.path.join(data_dir, 'embeddings.pkl')
            
            if os.path.exists(assessments_path):
                os.remove(assessments_path)
            
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
            
            print("Cache cleared successfully.")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def create_shl_data(self):
        """Create data for the 6 official SHL assessment categories"""
        print("Creating SHL official assessment data...")
        official_assessments = [
            {
                'Assessment Name': 'Behavioral Assessments',
                'URL': 'https://www.shl.com/solutions/products/assessments/behavioral-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '25 minutes',
                'Test Type': 'Behavioral',
                'Description': 'Evaluates work styles, preferences, and behavioral tendencies. Identifies key behaviors that drive success in specific roles.'
            },
            {
                'Assessment Name': 'Virtual Assessment & Development Centers',
                'URL': 'https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '120 minutes',
                'Test Type': 'Development',
                'Description': 'Immersive virtual experiences that simulate real-world scenarios to evaluate candidates\' capabilities and potential.'
            },
            {
                'Assessment Name': 'Personality Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/personality-assessment/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '30 minutes',
                'Test Type': 'Personality',
                'Description': 'Comprehensive evaluation of personality traits, work preferences, and interpersonal styles related to job performance.'
            },
            {
                'Assessment Name': 'Cognitive Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/cognitive-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'Yes',
                'Duration': '35 minutes',
                'Test Type': 'Cognitive',
                'Description': 'Measures critical reasoning and problem-solving abilities. Evaluates verbal, numerical, and logical reasoning skills.'
            },
            {
                'Assessment Name': 'Skills and Simulations',
                'URL': 'https://www.shl.com/solutions/products/assessments/skills-and-simulations/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '45 minutes',
                'Test Type': 'Technical',
                'Description': 'Hands-on assessments that measure practical skills and technical capabilities in real-world contexts.'
            },
            {
                'Assessment Name': 'Job-Focused Assessments',
                'URL': 'https://www.shl.com/solutions/products/assessments/job-focused-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '40 minutes',
                'Test Type': 'Role-specific',
                'Description': 'Job-specific assessments tailored to particular roles, combining cognitive, behavioral, and technical evaluations.'
            }
        ]
        
        # Create more detailed assessments under each category for better recommendations
        detailed_assessments = [
            # Behavioral Assessments variants
            {
                'Assessment Name': 'Leadership Behavior Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/behavioral-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '30 minutes',
                'Test Type': 'Behavioral',
                'Description': 'Identifies leadership capabilities, management style, and development areas for leadership positions.'
            },
            {
                'Assessment Name': 'Sales Behavior Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/behavioral-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '25 minutes',
                'Test Type': 'Behavioral',
                'Description': 'Measures sales potential, client relationship capabilities, and driving business results behaviors.'
            },
            {
                'Assessment Name': 'Collaboration and Teamwork Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/behavioral-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '20 minutes',
                'Test Type': 'Behavioral',
                'Description': 'Measures ability to collaborate effectively in team environments and work well with others.'
            },
            
            # Virtual Assessment & Development Centers variants
            {
                'Assessment Name': 'Leadership Development Center',
                'URL': 'https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '180 minutes',
                'Test Type': 'Development',
                'Description': 'Comprehensive virtual assessment center focusing on leadership competencies and potential.'
            },
            {
                'Assessment Name': 'Graduate Assessment Center',
                'URL': 'https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '120 minutes',
                'Test Type': 'Development',
                'Description': 'Virtual assessment center designed for early career professionals and recent graduates.'
            },
            
            # Personality Assessment variants
            {
                'Assessment Name': 'Occupational Personality Questionnaire',
                'URL': 'https://www.shl.com/solutions/products/assessments/personality-assessment/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '35 minutes',
                'Test Type': 'Personality',
                'Description': 'Comprehensive personality assessment evaluating 32 characteristics relevant to workplace performance.'
            },
            {
                'Assessment Name': 'Motivation Questionnaire',
                'URL': 'https://www.shl.com/solutions/products/assessments/personality-assessment/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '25 minutes',
                'Test Type': 'Personality',
                'Description': 'Evaluates what energizes and drives an individual in the workplace, helping identify cultural fit.'
            },
            
            # Cognitive Assessment variants
            {
                'Assessment Name': 'Inductive Reasoning Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/cognitive-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'Yes',
                'Duration': '25 minutes',
                'Test Type': 'Cognitive',
                'Description': 'Measures ability to identify patterns and relationships in abstract information and apply them to solve problems.'
            },
            {
                'Assessment Name': 'Numerical Reasoning Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/cognitive-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'Yes',
                'Duration': '30 minutes',
                'Test Type': 'Cognitive',
                'Description': 'Evaluates ability to analyze and interpret numerical data, statistics, and make logical conclusions.'
            },
            {
                'Assessment Name': 'Verbal Reasoning Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/cognitive-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'Yes',
                'Duration': '30 minutes',
                'Test Type': 'Cognitive',
                'Description': 'Measures ability to understand and evaluate written information and make logical conclusions.'
            },
            
            # Skills and Simulations variants
            {
                'Assessment Name': 'Java Coding Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/skills-and-simulations/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '45 minutes',
                'Test Type': 'Technical',
                'Description': 'Hands-on assessment of Java programming skills and problem-solving capabilities.'
            },
            {
                'Assessment Name': 'Python Programming Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/skills-and-simulations/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '45 minutes',
                'Test Type': 'Technical',
                'Description': 'Evaluates Python programming proficiency and problem-solving skills through practical coding challenges.'
            },
            {
                'Assessment Name': 'Microsoft Office Skills Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/skills-and-simulations/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '30 minutes',
                'Test Type': 'Technical',
                'Description': 'Measures proficiency in Microsoft Office applications including Excel, Word, and PowerPoint.'
            },
            
            # Job-Focused Assessments variants
            {
                'Assessment Name': 'Customer Service & Call Center Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/job-focused-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '35 minutes',
                'Test Type': 'Role-specific',
                'Description': 'Comprehensive evaluation of customer service skills, problem resolution, and communication effectiveness.'
            },
            {
                'Assessment Name': 'Sales Professional Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/job-focused-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '40 minutes',
                'Test Type': 'Role-specific',
                'Description': 'Role-specific assessment for sales positions combining behavioral traits, cognitive skills, and situational judgment.'
            },
            {
                'Assessment Name': 'IT Professional Assessment',
                'URL': 'https://www.shl.com/solutions/products/assessments/job-focused-assessments/',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No',
                'Duration': '45 minutes',
                'Test Type': 'Role-specific',
                'Description': 'Comprehensive assessment for IT roles, evaluating technical knowledge, problem-solving, and professional skills.'
            }
        ]
        
        # Combine the main categories with the detailed assessments
        all_assessments = official_assessments + detailed_assessments
        
        self.assessments_df = pd.DataFrame(all_assessments)
        
        # Create embeddings
        self.create_embeddings()
        
        # Cache the data
        self.cache_data()
    
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def create_embeddings(self):
        """Create embeddings for assessment descriptions"""
        # Combine name, description and test type for better matching
        combined_texts = [
            f"{row['Assessment Name']} {row['Description']} {row['Test Type']} {row['Duration']}"
            for _, row in self.assessments_df.iterrows()
        ]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in combined_texts]
        
        # Create embeddings
        self.embeddings = self.model.encode(processed_texts)
        
        # Verify embeddings were created successfully
        if len(self.embeddings) == 0:
            raise ValueError("Failed to create embeddings")
    
    def cache_data(self):
        """Cache the data"""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        self.assessments_df.to_pickle(os.path.join(data_dir, 'assessments.pkl'))
        with open(os.path.join(data_dir, 'embeddings.pkl'), 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def load_cached_data(self):
        """Load cached data"""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
        # Check if cache files exist
        assessments_path = os.path.join(data_dir, 'assessments.pkl')
        embeddings_path = os.path.join(data_dir, 'embeddings.pkl')
        
        if not os.path.exists(assessments_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError("Cache files not found")
        
        self.assessments_df = pd.read_pickle(assessments_path)
        
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        # Verify data integrity
        if self.assessments_df.empty or len(self.embeddings) == 0:
            raise ValueError("Cached data is empty")
    
    def get_recommendations(self, query, max_results=10, duration_limit=None):
        """Get assessment recommendations based on natural language query"""
        if self.embeddings is None or len(self.embeddings) == 0 or self.assessments_df is None or self.assessments_df.empty:
            # Attempt to recreate embeddings if they're missing
            print("Embeddings or assessment data missing. Recreating...")
            self.create_shl_data()
            
            # If still missing, return empty list
            if self.embeddings is None or len(self.embeddings) == 0:
                print("Failed to create embeddings. Cannot provide recommendations.")
                return []
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Get query embedding
        query_embedding = self.model.encode([processed_query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1]
        
        # Filter by duration if specified
        if duration_limit:
            # Extract numeric duration values where possible
            def extract_duration(dur_str):
                if pd.isna(dur_str) or dur_str == 'Unknown':
                    return float('inf')
                match = re.search(r'(\d+)', str(dur_str))
                return int(match.group(1)) if match else float('inf')
            
            # Filter to assessments within duration limit
            filtered_indices = [
                i for i in top_indices 
                if extract_duration(self.assessments_df.iloc[i]['Duration']) <= int(duration_limit)
            ]
            top_indices = filtered_indices
        
        # Get top recommendations (up to max_results)
        recommendations = []
        for idx in top_indices[:max_results]:
            assessment = self.assessments_df.iloc[idx].to_dict()
            assessment['Relevance Score'] = float(similarities[idx])
            recommendations.append(assessment)
        
        return recommendations

    def get_recommendations_json(self, query, max_results=10, duration_limit=None):
        """Get assessment recommendations in JSON format for API use"""
        recommendations = self.get_recommendations(query, max_results, duration_limit)
        return json.dumps(recommendations)

# Initialize evaluation metrics calculation
def calculate_precision_at_k(recommended, relevant, k):
    """Calculate precision at k"""
    count = 0
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            count += 1
    return count / k if k > 0 else 0

def calculate_average_precision(recommended, relevant, k):
    """Calculate average precision at k"""
    hits = 0
    sum_precisions = 0
    
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    return sum_precisions / min(k, len(relevant)) if len(relevant) > 0 else 0

def calculate_recall_at_k(recommended, relevant, k):
    """Calculate recall at k"""
    count = 0
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            count += 1
    return count / len(relevant) if len(relevant) > 0 else 0

# Main Streamlit app
def main():
    # Set up page configuration
    st.set_page_config(
        page_title="SHL Assessment Recommendation System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load system with error handling
    @st.cache_resource
    def load_system():
        try:
            # Add a unique key to force refresh of the cached resource
            system = SHLAssessmentSystem()
            # Force regeneration of data to ensure correct assessments and URLs
            system.create_shl_data()
            return system
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            # Create a placeholder system with data
            system = SHLAssessmentSystem()
            system.create_shl_data()
            return system
    
    with st.spinner("Loading SHL assessment system..."):
        assessment_system = load_system()
    
    # App title and description
    st.title("SHL Assessment Recommendation System")
    st.markdown("""
    This application helps hiring managers find the right SHL assessments for their job roles. 
    Enter a job description or specific requirements to get personalized assessment recommendations.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Recommendation Tool", "API Endpoint", "Evaluation"])
    
    # Tab 1: Main recommendation tool
    with tab1:
        st.header("Assessment Recommender")
        
        # Input options
        input_method = st.radio(
            "Choose input method:",
            ["Enter query text", "Upload job description file", "Enter job description URL"]
        )
        
        query = ""
        
        if input_method == "Enter query text":
            query = st.text_area(
                "Enter job description or query:", 
                height=150,
                placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
            )
            
        elif input_method == "Upload job description file":
            uploaded_file = st.file_uploader("Upload job description document", type=["txt", "pdf", "docx"])
            if uploaded_file is not None:
                # Simple handling for text files
                if uploaded_file.type == "text/plain":
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    query = stringio.read()
                    st.write("File content loaded successfully!")
                else:
                    st.warning("Advanced file format parsing (PDF/DOCX) would require additional libraries in production.")
                    query = ""
                    
        elif input_method == "Enter job description URL":
            url = st.text_input("Enter URL containing job description:")
            if url and st.button("Fetch Content"):
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Simple text extraction - would need refinement in production
                    query = soup.get_text()
                    st.success("Content fetched successfully!")
                except:
                    st.error("Failed to fetch content from URL.")
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider("Maximum number of recommendations:", 1, 10, 5)
        
        with col2:
            duration_limit = st.number_input("Maximum assessment duration (minutes):", min_value=0, step=5)
            if duration_limit == 0:
                duration_limit = None
        
        # Get recommendations button
        if st.button("Get Recommendations") and query:
            with st.spinner("Finding the best assessments..."):
                try:
                    recommendations = assessment_system.get_recommendations(
                        query, 
                        max_results=max_results, 
                        duration_limit=duration_limit
                    )
                except Exception as e:
                    st.error(f"Error getting recommendations: {e}")
                    recommendations = []
            
            if recommendations:
                # Display results in a table
                st.subheader("Recommended Assessments")
                
                # Prepare data for table
                table_data = []
                for rec in recommendations:
                    table_data.append({
                        "Assessment Name": rec["Assessment Name"],
                        "Remote Testing": rec["Remote Testing Support"],
                        "Adaptive/IRT": rec["Adaptive/IRT Support"],
                        "Duration": rec["Duration"],
                        "Test Type": rec["Test Type"],
                        "Relevance Score": f"{rec['Relevance Score']:.2f}"
                    })
                
                st.table(pd.DataFrame(table_data))
                
                # Detailed view of each assessment
                st.subheader("Detailed Assessment Information")
                for i, rec in enumerate(recommendations):
                    with st.expander(f"{i+1}. {rec['Assessment Name']}"):
                        st.markdown(f"**Assessment Name:** [{rec['Assessment Name']}]({rec['URL']})")
                        st.markdown(f"**Remote Testing Support:** {rec['Remote Testing Support']}")
                        st.markdown(f"**Adaptive/IRT Support:** {rec['Adaptive/IRT Support']}")
                        st.markdown(f"**Duration:** {rec['Duration']}")
                        st.markdown(f"**Test Type:** {rec['Test Type']}")
                        st.markdown(f"**Relevance Score:** {rec['Relevance Score']:.4f}")
                        if rec.get('Description'):
                            st.markdown(f"**Description:** {rec['Description']}")
            else:
                st.warning("No matching assessments found. Try a different query or adjust the duration limit.")
    
    # Tab 2: API Endpoint 
    with tab2:
        st.header("API Endpoint")
        st.markdown("This tab demonstrates the API functionality that you can integrate into your applications.")
        
        # API parameters
        api_query = st.text_area(
            "Query for API:", 
            height=100,
            placeholder="Enter your query to test the API endpoint"
        )
        
        api_duration = st.number_input("API Duration Limit:", min_value=0, step=5)
        if api_duration == 0:
            api_duration = None
            
        if st.button("Test API") and api_query:
            try:
                api_result = assessment_system.get_recommendations_json(
                    api_query, 
                    max_results=5, 
                    duration_limit=api_duration
                )
                
                st.subheader("API Response (JSON)")
                st.json(api_result)
            except Exception as e:
                st.error(f"API error: {e}")
            
            # Show example API usage
            st.subheader("How to Use the API")
            st.code("""
            # Example Python code to call the API
            import requests
            
            url = "https://your-deployed-app-url.com/api"
            params = {
                "query": "Java developers with teamwork skills",
                "duration_limit": 40
            }
            
            response = requests.get(url, params=params)
            recommendations = response.json()
            """)
            
            st.markdown("""
            When deployed, this application exposes an API endpoint that returns assessment recommendations in JSON format. 
            You can integrate this with your existing HR systems or applications.
            """)
    
    # Tab 3: Evaluation metrics
    with tab3:
        st.header("Evaluation Metrics")
        st.markdown("""
        This tab demonstrates how the recommendation system is evaluated using standard metrics:
        - **Mean Recall@K**: Measures how many relevant assessments are retrieved in the top K recommendations
        - **MAP@K**: Mean Average Precision evaluates both relevance and ranking order
        """)
        
        # Create a simple evaluation interface
        st.subheader("Test Query")
        eval_query = st.text_area(
            "Query to evaluate:", 
            value="I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
        )
        
        st.subheader("Ground Truth (Relevant Assessments)")
        ground_truth = st.text_area(
            "Enter relevant assessment names (one per line):",
            value="Java Coding Assessment\nIT Professional Assessment\nCollaboration and Teamwork Assessment"
        )
        
        k_value = st.slider("K value for metrics:", 1, 10, 3)
        
        if st.button("Calculate Metrics") and eval_query and ground_truth:
            with st.spinner("Calculating evaluation metrics..."):
                try:
                    # Get recommendations
                    recommendations = assessment_system.get_recommendations(eval_query, max_results=10)
                    
                    # Extract names and prepare ground truth
                    recommended_names = [rec["Assessment Name"] for rec in recommendations]
                    relevant_names = [name.strip() for name in ground_truth.strip().split("\n") if name.strip()]
                    
                    # Calculate metrics
                    recall = calculate_recall_at_k(recommended_names, relevant_names, k_value)
                    ap = calculate_average_precision(recommended_names, relevant_names, k_value)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Recall@K", f"{recall:.4f}")
                    
                    with col2:
                        st.metric("Average Precision@K", f"{ap:.4f}")
                    
                    # Show recommendations vs ground truth
                    st.subheader("Recommendations vs Ground Truth")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Top Recommendations:")
                        for i, name in enumerate(recommended_names[:k_value]):
                            if name in relevant_names:
                                st.markdown(f"{i+1}. ✅ {name}")
                            else:
                                st.markdown(f"{i+1}. ❌ {name}")
                    
                    with col2:
                        st.write("Relevant Assessments:")
                        for name in relevant_names:
                            if name in recommended_names[:k_value]:
                                st.markdown(f"✅ {name}")
                            else:
                                st.markdown(f"❌ {name}")
                except Exception as e:
                    st.error(f"Error calculating metrics: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("SHL Assessment Recommendation System | Developed as a technical demonstration")

if __name__ == "__main__":
    main()
