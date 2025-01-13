import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter, defaultdict
import nltk
import re
import os
from datetime import datetime
import traceback

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class RequirementAnalyzer:
    def __init__(self):
        self.requirement_categories = {
            'hardware': ['sensor', 'camera', 'radar', 'lidar', 'hardware', 'device', 'equipment', 
                        'computing', 'processor', 'gpu', 'system', 'interface', 'screen', 'display'],
            'software': ['software', 'program', 'application', 'platform', 'system', 'interface', 
                        'api', 'algorithm', 'model', 'simulation', 'framework'],
            'data': ['data', 'dataset', 'information', 'measurement', 'recording', 'sample', 
                    'collection', 'storage', 'database', 'stream', 'feed'],
            'infrastructure': ['network', 'connection', 'infrastructure', 'facility', 'environment', 
                             'space', 'area', 'room', 'field', 'setup', 'installation'],
            'performance': ['speed', 'accuracy', 'precision', 'resolution', 'quality', 'performance', 
                          'efficiency', 'reliability', 'robustness', 'capability'],
            'integration': ['integration', 'compatibility', 'interface', 'connection', 'communication', 
                          'interaction', 'synchronization', 'coordination'],
            'operational': ['operation', 'procedure', 'process', 'workflow', 'protocol', 'standard', 
                          'requirement', 'specification', 'guideline']
        }
        
        self.requirement_indicators = [
            'need', 'require', 'must', 'should', 'want', 'necessary', 'essential', 'important',
            'critical', 'crucial', 'specific', 'particular', 'custom', 'precise'
        ]

    def extract_requirements(self, text):
        if pd.isna(text):
            return {}
        
        text = str(text).lower()
        sentences = sent_tokenize(text)
        requirements = defaultdict(list)
        
        for sentence in sentences:
            if any(indicator in sentence for indicator in self.requirement_indicators):
                for category, keywords in self.requirement_categories.items():
                    if any(keyword in sentence for keyword in keywords):
                        requirements[category].append(sentence.strip())
            else:
                # Also check for implicit requirements (sentences with keywords but without explicit indicators)
                for category, keywords in self.requirement_categories.items():
                    if any(keyword in sentence for keyword in keywords):
                        requirements[category].append(sentence.strip())
        
        return requirements

def preprocess_text(text):
    """Preprocess text by removing special characters, converting to lowercase, and lemmatizing words"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_key_phrases(texts, n_topics=3, n_words=8):
    """Extract key themes using Latent Dirichlet Allocation"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    texts = [text for text in texts if text.strip()]
    if not texts:
        return ["No valid text found for analysis"]
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(tfidf_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        themes = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            themes.append(f"Theme {topic_idx + 1}: {', '.join(top_words)}")
        
        return themes
    except Exception as e:
        return [f"Error in theme extraction: {str(e)}"]

def analyze_patterns(texts):
    """Analyze common patterns in the texts"""
    all_words = []
    bigrams = []
    key_phrases = []
    
    for text in texts:
        if pd.isna(text):
            continue
            
        tokens = word_tokenize(str(text).lower())
        tagged = pos_tag(tokens)
        
        all_words.extend([word for word, tag in tagged if tag.startswith(('NN', 'VB', 'JJ'))])
        
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")
        
        for i in range(len(tagged) - 1):
            if tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN'):
                key_phrases.append(f"{tagged[i][0]} {tagged[i+1][0]}")
    
    return {
        'common_words': Counter(all_words).most_common(10),
        'common_bigrams': Counter(bigrams).most_common(10),
        'key_phrases': Counter(key_phrases).most_common(10)
    }

def analyze_needs(responses):
    """Analyze and summarize needs from responses"""
    analyzer = RequirementAnalyzer()
    all_requirements = defaultdict(list)
    
    for response in responses:
        if pd.isna(response):
            continue
        
        requirements = analyzer.extract_requirements(response)
        for category, reqs in requirements.items():
            all_requirements[category].extend(reqs)
    
    # Summarize requirements
    summary = {}
    for category, reqs in all_requirements.items():
        if reqs:
            unique_reqs = list(dict.fromkeys(reqs))
            summary[category] = unique_reqs
    
    return summary

def write_to_file(filename, content):
    """Write content to a file with proper encoding"""
    filepath = os.path.join(r"C:\Users\me\OneDrive\Documents\RCODI\AIDA3", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def format_analysis_output(column_name, needs_summary, themes, patterns):
    """Format the analysis results for file output"""
    output = []
    output.append("="*80)
    output.append(f"Analysis for: {column_name}")
    output.append("="*80)
    
    output.append("\nIDENTIFIED NEEDS AND REQUIREMENTS")
    output.append("-"*50)
    for category, requirements in needs_summary.items():
        output.append(f"\n{category.title()} Requirements:")
        for req in requirements:
            output.append(f"- {req}")
    
    output.append("\nKEY THEMES")
    output.append("-"*50)
    for theme in themes:
        output.append(theme)
    
    output.append("\nCOMMON TERMS")
    output.append("-"*50)
    for word, count in patterns['common_words']:
        output.append(f"- {word}: {count} occurrences")
    
    return "\n".join(output)

def main():
    try:
        # Read the CSV file
        df = pd.read_csv(r"C:\Users\me\OneDrive\Documents\RCODI\AIDA3\UseCaseSurvey.csv")
        
        # Create timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define the columns to analyze
        columns_to_analyze = [
            'How would you like to use PURT remotely? Do you have specific requirements concerning such an UAS indoor motion capture lab? Do you have specific requirements with respect to emulating certain communication and networking conditions or other real-world environmental conditions (e.g. GNSS signal degradation, wind, urban canyons, etc.)?',
            'How would you like to use the SOC? (e.g. Do you have any specific requirements in terms of sensors? Do you have any specific requirements in terms of VR/AR technologies? How large is the team size you want to work with? Do you have specific requirements in terms of sensors to perform research using real-time sensing of human cognition (e.g. eyetracking etc.) and brain-computer interfaces and neuro-inspired human-robot-interaction (e.g. biofeedback, neurofeedback)? What requirements do you have with respect to our motion-capture systems and cameras?)',
            'How would you like to use the PUC? What are your specific requirements in terms of motion-capture, urban infrastructures (e.g. materials, etc.)? What forms of signal degradation do you want to study and how?',
            'How would you like to use the PUP Airfield? Are there specific requirements in onsite equipment and sensors? Besides collecting ground truth data, what are data matter?',
            'How would you like to use our fleet? (e.g. What vehicles would you like to use and how? Do you have specific requirements in terms of onboard sensing (e.g. SAR, airborne radars, LiDAR, thermal and hyperspectral, etc) or onboard computing resources? What vehicles would you like to study and how? Do you want to bring your own vehicles?)',
            'How would you like to use Purdue XTM? (e.g. are their specific radar requirements? Is there a need for portable radars? What are particular requirements with respect to low altitude weather network station? Where do you want to place them?)',
            'How would you like to use a digital twin? Do have a specific physical system in mind for which you need high fidelity simulation models? How would you like to use such digital twins (e.g. simulations and counterfactual reasoning etc.)'
        ]

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        columns_to_analyze = [col.strip() for col in columns_to_analyze]
        
        # Create summary file for all analyses
        summary_content = []
        
        # Analyze each column
        for column in columns_to_analyze:
            print(f"\nAnalyzing: {column[:50]}...")  # Print truncated column name
            
            if column in df.columns:
                # Get responses and analyze them
                responses = df[column].dropna().astype(str)
                if len(responses) == 0:
                    print("No responses found for this question")
                    continue
                
                # Perform analyses
                needs_summary = analyze_needs(responses)
                processed_texts = [preprocess_text(text) for text in responses]
                themes = extract_key_phrases(processed_texts)
                patterns = analyze_patterns(responses)
                
                # Format output
                analysis_output = format_analysis_output(
                    column,
                    needs_summary,
                    themes,
                    patterns
                )
                
                # Add to summary
                summary_content.append(analysis_output)
                
                # Create individual file for this analysis
                facility_name = re.search(r'use\s+(\w+)', column)
                filename = f"analysis_{facility_name.group(1) if facility_name else 'unnamed'}_{timestamp}.txt"
                write_to_file(filename, analysis_output)
                print(f"Written analysis to {filename}")
            
            else:
                print(f"Column not found in CSV")
        
        # Write combined analyses to summary file
        summary_filename = f"complete_analysis_summary_{timestamp}.txt"
        write_to_file(summary_filename, "\n\n".join(summary_content))
        print(f"\nComplete analysis written to {summary_filename}")

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        write_to_file(f"error_log_{timestamp}.txt", error_msg)

if __name__ == "__main__":
    main()
