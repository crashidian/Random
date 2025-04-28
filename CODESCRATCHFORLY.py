import os
import json
import csv
import nltk
import tkinter as tk
from tkinter import filedialog
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('brown')

class TemporalNLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.temporal_words = self._get_temporal_words()
        self.stop_words.update(self.temporal_words)
        self.irregular_verbs = self._get_irregular_verbs()
        
    def _get_temporal_words(self):
        """Get temporal words for analysis"""
        # Start with seed temporal words
        temporal_words = {'now', 'then', 'before', 'after', 'while', 'during', 
                          'since', 'until', 'when', 'whenever', 'always', 'never',
                          'sometimes', 'often', 'rarely', 'today', 'tomorrow', 
                          'yesterday', 'will', 'shall', 'would', 'should', 'can',
                          'could', 'may', 'might', 'must'}
        
        # Add time-related words from WordNet
        for synset in wn.synsets('time', pos=wn.NOUN):
            for lemma in synset.lemmas():
                temporal_words.add(lemma.name().lower())
            
            # Get hyponyms (more specific time concepts)
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    temporal_words.add(lemma.name().lower())
        
        return temporal_words
        
    def _get_irregular_verbs(self):
        """Simplified list of common irregular verbs"""
        return {
            'be': {'base': 'be', 'past': 'was', 'past_participle': 'been', 'present_3sg': 'is', 'present_participle': 'being'},
            'have': {'base': 'have', 'past': 'had', 'past_participle': 'had', 'present_3sg': 'has', 'present_participle': 'having'},
            'do': {'base': 'do', 'past': 'did', 'past_participle': 'done', 'present_3sg': 'does', 'present_participle': 'doing'},
            'go': {'base': 'go', 'past': 'went', 'past_participle': 'gone', 'present_3sg': 'goes', 'present_participle': 'going'},
            'say': {'base': 'say', 'past': 'said', 'past_participle': 'said', 'present_3sg': 'says', 'present_participle': 'saying'},
        }
    
    def get_verb_form_and_tense(self, word):
        """Identify the form and tense of a verb"""
        # Check if this is an irregular verb we know
        for verb_info in self.irregular_verbs.values():
            if word.lower() == verb_info['past']:
                return (verb_info['base'], 'PAST')
            elif word.lower() == verb_info['past_participle']:
                return (verb_info['base'], 'PAST_PARTICIPLE')
            elif word.lower() == verb_info['present_3sg']:
                return (verb_info['base'], 'PRESENT')
            elif word.lower() == verb_info['present_participle']:
                return (verb_info['base'], 'PRESENT_CONTINUOUS')
        
        # For regular verbs or unknown irregulars
        lemma = self.lemmatizer.lemmatize(word, 'v')
        
        if word != lemma:
            if word.endswith('ed'):
                return (lemma, 'PAST')
            elif word.endswith('ing'):
                return (lemma, 'PRESENT_CONTINUOUS')
            elif word.endswith('s'):
                return (lemma, 'PRESENT')
        
        # Default case
        return (lemma, 'PRESENT_BASE')
    
    def get_wordnet_pos(self, treebank_tag):
        """Map POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    def select_file(self):
        """Allow user to select a file"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[
                ("All Supported Files", "*.json;*.csv;*.txt"),
                ("JSON Files", "*.json"),
                ("CSV Files", "*.csv"),
                ("Text Files", "*.txt")
            ]
        )
        
        if not file_path:
            print("No file selected. Exiting.")
            return None
            
        return file_path
    
    def read_file(self, file_path):
        """Read content from various file formats"""
        _, file_extension = os.path.splitext(file_path)
        
        try:
            if file_extension.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'text' in data:
                    return data['text']
                elif isinstance(data, list) and all('text' in item for item in data):
                    return ' '.join(item['text'] for item in data)
                else:
                    return str(data)
                    
            elif file_extension.lower() == '.csv':
                texts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        texts.append(' '.join(row))
                return ' '.join(texts)
                
            elif file_extension.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
                
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def analyze_text(self, text):
        """Analyze text for temporal logic validity"""
        sentences = sent_tokenize(text)
        results = []
        
        for sentence in sentences:
            # Extract words and filter stopwords
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            
            # POS tagging with error handling
            try:
                pos_tags = pos_tag(filtered_words)
            except LookupError:
                print("Warning: POS tagger not available. Using simple default tagging.")
                # Simple fallback tagging
                pos_tags = [(word, 'NN' if word[0].isupper() else 'VB') for word in filtered_words]
            
            # Identify subjects, verbs, and temporal markers
            subjects = []
            verbs = []
            temporal_markers = []
            
            for word, tag in pos_tags:
                wn_pos = self.get_wordnet_pos(tag)
                
                if wn_pos == wn.NOUN:
                    subjects.append(word)
                elif wn_pos == wn.VERB:
                    base_form, tense = self.get_verb_form_and_tense(word)
                    verbs.append({'word': word, 'base': base_form, 'tense': tense})
                elif word.lower() in self.temporal_words:
                    temporal_markers.append(word)
            
            # Evaluate temporal logic consistency
            is_valid = True
            reason = "Valid temporal statement"
            
            if verbs:
                main_verb = verbs[0]
                tense = main_verb['tense']
                
                # Check for contradictions
                if tense == 'PAST' and any(m in ('will', 'shall') for m in temporal_markers):
                    is_valid = False
                    reason = "Contradiction: Past tense with future modal verb"
                
                if tense == 'PRESENT_CONTINUOUS' and any(m in ('would', 'will') for m in temporal_markers):
                    is_valid = False
                    reason = "Contradiction: Present continuous with would/will"
                    
                # Check for completion verbs in continuous tense
                completion_verbs = {'complete', 'finish', 'end', 'conclude', 'terminate'}
                if main_verb['base'] in completion_verbs and tense == 'PRESENT_CONTINUOUS':
                    is_valid = False
                    reason = f"Invalid: '{main_verb['base']}' implies completion and cannot be ongoing"
                
                # Generate temporal logic formula
                if subjects:
                    subject = subjects[0]
                    verb = main_verb['base']
                    obj = subjects[1] if len(subjects) > 1 else "X"
                    
                    formula = f"{tense}({subject}, {verb}, {obj})"
                    
                    # Generate response
                    if is_valid:
                        response = f"Task can be completed: '{subject} {main_verb['word']} {obj}'."
                    else:
                        response = f"Task cannot be completed: {reason}."
                    
                    results.append({
                        'sentence': sentence,
                        'formula': formula,
                        'is_valid': is_valid,
                        'reason': reason,
                        'response': response
                    })
        
        return results
    
    def run(self):
        """Run the complete processing pipeline"""
        # Step 1: Select file
        file_path = self.select_file()
        if not file_path:
            return
            
        # Step 2: Read file content
        text = self.read_file(file_path)
        if not text:
            return
            
        print(f"Processing file: {file_path}")
        
        # Step 3: Analyze text
        results = self.analyze_text(text)
        
        # Step 4: Print results
        print("\n=== Temporal Logic Analysis ===")
        for result in results:
            status = "VALID" if result['is_valid'] else "INVALID"
            print(f"- [{status}] {result['formula']}")
            print(f"  Reason: {result['reason']}")
            print(f"  Response: {result['response']}")
            print()
        
        return results


# Main execution
if __name__ == "__main__":
    processor = TemporalNLPProcessor()
    processor.run()
