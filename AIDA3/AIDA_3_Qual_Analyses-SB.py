import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from collections import Counter, defaultdict
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class RequirementAnalyzer:
    def __init__(self):
        # Define requirement categories and their associated keywords
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
        
        # Keywords indicating requirements
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
            # Check if sentence contains requirement indicators
            if any(indicator in sentence for indicator in self.requirement_indicators):
                # Categorize the requirement
                for category, keywords in self.requirement_categories.items():
                    if any(keyword in sentence for keyword in keywords):
                        requirements[category].append(sentence.strip())
        
        return requirements

def preprocess_text(text):
    """
    Preprocess text by removing special characters, converting to lowercase,
    and lemmatizing words
    """
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
    """
    Extract key themes using Latent Dirichlet Allocation
    """
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
    """
    Analyze common patterns in the texts
    """
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
    """
    Analyze and summarize needs from responses
    """
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
            # Remove duplicates while preserving order
            unique_reqs = list(dict.fromkeys(reqs))
            summary[category] = unique_reqs
    
    return summary

def main():
    try:
        # Read the CSV file
        df = pd.read_csv(r"UseCaseSurvey 1.csv")
        
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
        
        # Analyze each column
        for column in columns_to_analyze:
            print(f"\nAnalyzing responses for: {column}")
            print("-" * 50)
            
            if column in df.columns:
                responses = df[column].dropna().astype(str)
                if len(responses) == 0:
                    print("No responses found for this question")
                    continue
                
                # Extract and analyze needs
                print("\nIdentified Needs and Requirements:")
                needs_summary = analyze_needs(responses)
                for category, requirements in needs_summary.items():
                    print(f"\n{category.title()} Requirements:")
                    for req in requirements:
                        print(f"- {req}")
                
                # Extract themes
                print("\nKey Themes:")
                processed_texts = [preprocess_text(text) for text in responses]
                themes = extract_key_phrases(processed_texts)
                for theme in themes:
                    print(theme)
                
                # Analyze patterns
                patterns = analyze_patterns(responses)
                
                print("\nMost Common Terms:")
                for word, count in patterns['common_words']:
                    print(f"- {word}: {count} occurrences")
                
                print("\n" + "="*80 + "\n")
            else:
                print(f"Column not found in CSV")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
