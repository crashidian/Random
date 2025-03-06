import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from neo4j import GraphDatabase

# Download basic NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

# Common English stopwords
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
}

# Simple POS tagger - no complex imports needed
def simple_pos_tag(tokens):
    tagged = []
    for token in tokens:
        if token.lower() in STOP_WORDS:
            tag = "DT"  # Determiner
        elif token.endswith('ing'):
            tag = "VBG"  # Verb gerund
        elif token.endswith('ed'):
            tag = "VBD"  # Verb past tense
        elif token.endswith('ly'):
            tag = "RB"   # Adverb
        elif token.endswith('s') and not token.endswith(('ss', 'us', 'is')):
            tag = "NNS"  # Plural noun
        elif token[0].isupper() and token not in ['I', 'A']:
            tag = "NNP"  # Proper noun
        else:
            tag = "NN"   # Noun (default)
        tagged.append((token, tag))
    return tagged

class TranscriptParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.transcript_text = self._read_file()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize data structures
        self.speakers = {}
        self.utterances = []
        self.entities = []
        self.topics = []
        self.relationships = []
        self.activities = []
        self.temporal_markers = []
        
        # Activity verbs
        self.activity_verbs = [
            'teach', 'work', 'drive', 'fix', 'eat', 'make', 'cook', 
            'live', 'talk', 'speak', 'think', 'remember', 'forget'
        ]
        
        print(f"Initialized parser for file: {file_path}")
    
    def _read_file(self):
        """Read the transcript file with error handling"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"Successfully read file: {len(content)} characters")
                return content
        except UnicodeDecodeError:
            try:
                with open(self.file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    print(f"Successfully read file with latin-1 encoding: {len(content)} characters")
                    return content
            except Exception as e:
                print(f"Error reading file with latin-1 encoding: {e}")
                return ""
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def split_into_sentences(self, text):
        """Split text into sentences using regex"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text):
        """Tokenize text into words using regex"""
        try:
            return word_tokenize(text)
        except:
            # Fallback tokenizer
            return re.findall(r'\b\w+(?:\'t|\'s|\'m|\'re|\'ll|\'ve|\'d)?\b', text.lower())
    
    def is_numeric(self, text):
        """Check if a string is numeric"""
        if not text:
            return False
        # Convert to string and check if it's numeric
        text = str(text).replace(',', '').replace('.', '').replace('-', '')
        return text.isdigit() or text.replace('.', '', 1).isdigit()
    
    def parse_speakers(self):
        """Extract speakers from the transcript"""
        print("Parsing speakers from transcript")
        speaker_pattern = re.compile(r'(\d+:\d+:\d+)\s+(Speaker\s+\d+)')
        matches = speaker_pattern.findall(self.transcript_text)
        
        for time, speaker in matches:
            if speaker not in self.speakers:
                self.speakers[speaker] = []
            self.speakers[speaker].append(time)
            
        print(f"Found {len(self.speakers)} distinct speakers")
        return self.speakers
    
    def extract_utterances(self):
        """Extract utterances with speaker information"""
        print("Extracting utterances from transcript")
        lines = self.transcript_text.split('\n')
        
        current_speaker = None
        current_utterance = ""
        
        for line in lines:
            speaker_match = re.search(r'(\d+:\d+:\d+)\s+(Speaker\s+\d+)', line)
            if speaker_match:
                # Save the previous utterance
                if current_speaker and current_utterance.strip():
                    self.utterances.append({
                        "speaker": current_speaker,
                        "text": current_utterance.strip(),
                        "time": speaker_match.group(1)
                    })
                
                # Start a new utterance
                current_speaker = speaker_match.group(2)
                current_utterance = line[speaker_match.end():].strip()
            else:
                # Continue the current utterance
                current_utterance += " " + line.strip()
        
        # Add the last utterance
        if current_speaker and current_utterance.strip():
            self.utterances.append({
                "speaker": current_speaker,
                "text": current_utterance.strip(),
                "time": "end"
            })
            
        print(f"Extracted {len(self.utterances)} utterances")
        return self.utterances
    
    def extract_entities(self):
        """Extract entities from the transcript"""
        print("Extracting entities from transcript")
        entity_dict = {}
        
        for utterance in self.utterances:
            sentences = self.split_into_sentences(utterance["text"])
            
            for sentence in sentences:
                # Tokenize and tag words
                words = self.tokenize_words(sentence)
                tagged = simple_pos_tag(words)
                
                # Find proper nouns and noun phrases
                i = 0
                while i < len(tagged):
                    word, tag = tagged[i]
                    
                    # Look for proper nouns (NNP)
                    if tag == "NNP":
                        # Build multi-word entities
                        entity_parts = [word]
                        j = i + 1
                        while j < len(tagged) and tagged[j][1] == "NNP":
                            entity_parts.append(tagged[j][0])
                            j += 1
                        
                        entity_name = ' '.join(entity_parts)
                        
                        # Skip numeric entities
                        if not self.is_numeric(entity_name):
                            if entity_name not in entity_dict:
                                entity_dict[entity_name] = {
                                    "name": entity_name,
                                    "type": "PERSON",
                                    "mentions": 1,
                                    "context": sentence,
                                    "speaker": utterance["speaker"]
                                }
                            else:
                                entity_dict[entity_name]["mentions"] += 1
                                
                        i = j  # Skip ahead
                    else:
                        i += 1
                
                # Also look for important nouns
                for i in range(len(tagged)):
                    if tagged[i][1] == "NN" and not self.is_numeric(tagged[i][0]) and len(tagged[i][0]) > 3:
                        word = tagged[i][0]
                        if word not in entity_dict and word not in STOP_WORDS:
                            entity_dict[word] = {
                                "name": word,
                                "type": "CONCEPT",
                                "mentions": 1,
                                "context": sentence,
                                "speaker": utterance["speaker"]
                            }
                        elif word in entity_dict:
                            entity_dict[word]["mentions"] += 1
        
        # Convert to list and filter out low-mention entities
        self.entities = [e for e in entity_dict.values() if e["mentions"] > 1]
        
        # Sort by mentions
        self.entities = sorted(self.entities, key=lambda x: x["mentions"], reverse=True)
        
        print(f"Found {len(self.entities)} entities")
        return self.entities
    
    def extract_topics(self):
        """Extract topics from the transcript"""
        print("Extracting topics from transcript")
        word_counts = {}
        word_contexts = {}
        
        for utterance in self.utterances:
            # Split into words
            words = self.tokenize_words(utterance["text"].lower())
            
            # Count lemmatized content words
            for word in words:
                if (word not in STOP_WORDS and 
                    len(word) > 3 and 
                    not self.is_numeric(word)):
                    
                    lemma = self.lemmatizer.lemmatize(word)
                    
                    # Count frequency
                    if lemma in word_counts:
                        word_counts[lemma] += 1
                    else:
                        word_counts[lemma] = 1
                    
                    # Store context
                    sentences = self.split_into_sentences(utterance["text"])
                    for sentence in sentences:
                        if word in sentence.lower():
                            if lemma not in word_contexts:
                                word_contexts[lemma] = []
                            if sentence not in word_contexts[lemma]:
                                word_contexts[lemma].append(sentence)
        
        # Convert to list of topics
        for word, count in word_counts.items():
            if count > 1:  # Only include topics mentioned more than once
                context = word_contexts.get(word, [""])[0]  # Use first context
                
                self.topics.append({
                    "word": word,
                    "frequency": count,
                    "context": context
                })
        
        # Sort by frequency
        self.topics = sorted(self.topics, key=lambda x: x["frequency"], reverse=True)
        
        # Limit to top 100 topics
        self.topics = self.topics[:100]
        
        print(f"Extracted {len(self.topics)} topics")
        return self.topics
    
    def identify_relationships(self):
        """Identify relationships between entities"""
        print("Identifying relationships between entities")
        entity_names = [e["name"].lower() for e in self.entities]
        
        for utterance in self.utterances:
            sentences = self.split_into_sentences(utterance["text"])
            
            for sentence in sentences:
                # Find entities mentioned in this sentence
                mentioned_entities = []
                for entity in self.entities:
                    if entity["name"].lower() in sentence.lower():
                        mentioned_entities.append(entity["name"])
                
                # Create co-occurrence relationships between entities
                for i in range(len(mentioned_entities)):
                    for j in range(i+1, len(mentioned_entities)):
                        self.relationships.append({
                            "subject": mentioned_entities[i],
                            "predicate": "mentioned_with",
                            "object": mentioned_entities[j],
                            "speaker": utterance["speaker"],
                            "confidence": 0.7,
                            "context": sentence
                        })
                
                # Also look for simple subject-verb-object patterns
                words = self.tokenize_words(sentence)
                tagged = simple_pos_tag(words)
                
                for i in range(len(tagged) - 2):
                    if ((tagged[i][1] in ["NN", "NNP", "NNS"]) and
                        (tagged[i+1][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) and
                        (tagged[i+2][1] in ["NN", "NNP", "NNS"])):
                        
                        subject = tagged[i][0]
                        verb = tagged[i+1][0]
                        obj = tagged[i+2][0]
                        
                        # Skip if any part is numeric
                        if (self.is_numeric(subject) or 
                            self.is_numeric(verb) or 
                            self.is_numeric(obj)):
                            continue
                        
                        self.relationships.append({
                            "subject": subject,
                            "predicate": verb,
                            "object": obj,
                            "speaker": utterance["speaker"],
                            "confidence": 0.5,
                            "context": sentence
                        })
        
        print(f"Identified {len(self.relationships)} relationships")
        return self.relationships
    
    def extract_activities(self):
        """Extract activities from the transcript"""
        print("Extracting activities from transcript")
        for utterance in self.utterances:
            sentences = self.split_into_sentences(utterance["text"])
            
            for sentence in sentences:
                words = self.tokenize_words(sentence.lower())
                
                for i, word in enumerate(words):
                    lemma = self.lemmatizer.lemmatize(word)
                    
                    if lemma in self.activity_verbs:
                        # Found an activity verb
                        self.activities.append({
                            "activity": lemma,
                            "context": sentence,
                            "speaker": utterance["speaker"]
                        })
        
        print(f"Extracted {len(self.activities)} activities")
        return self.activities
    
    def extract_temporal_markers(self):
        """Extract temporal markers from the transcript"""
        print("Extracting temporal markers from transcript")
        time_words = ['today', 'tomorrow', 'yesterday', 'now', 'then', 'before', 'after', 
                     'while', 'during', 'year', 'month', 'week', 'day', 'time']
        
        for utterance in self.utterances:
            sentences = self.split_into_sentences(utterance["text"])
            
            for sentence in sentences:
                words = self.tokenize_words(sentence.lower())
                
                for word in words:
                    if word in time_words:
                        self.temporal_markers.append({
                            "marker": word,
                            "text": sentence,
                            "speaker": utterance["speaker"]
                        })
        
        print(f"Found {len(self.temporal_markers)} temporal markers")
        return self.temporal_markers
    
    def build_knowledge_graph(self):
        """Build a complete knowledge graph representation"""
        print("\nBuilding knowledge graph...")
        
        print("Parsing speakers...")
        speakers = self.parse_speakers()
        
        print("Extracting utterances...")
        utterances = self.extract_utterances()
        
        print("Extracting entities...")
        entities = self.extract_entities()
        
        print("Extracting topics...")
        topics = self.extract_topics()
        
        print("Identifying relationships...")
        relationships = self.identify_relationships()
        
        print("Extracting activities...")
        activities = self.extract_activities()
        
        print("Extracting temporal markers...")
        temporal_markers = self.extract_temporal_markers()
        
        knowledge_graph = {
            "speakers": speakers,
            "utterances": utterances,
            "entities": entities,
            "topics": topics,
            "relationships": relationships,
            "activities": activities,
            "temporal_markers": temporal_markers
        }
        
        return knowledge_graph


class Neo4jConnector:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
    
    def create_speaker_nodes(self, speakers):
        """Create Speaker nodes"""
        with self.driver.session() as session:
            for speaker, times in speakers.items():
                query = (
                    "MERGE (s:Speaker {name: $name}) "
                    "SET s.utterance_count = $count "
                    "RETURN s"
                )
                session.run(query, name=speaker, count=len(times))
    
    def create_entity_nodes(self, entities):
        """Create Entity nodes"""
        with self.driver.session() as session:
            for entity in entities:
                query = (
                    "MERGE (e:Entity {name: $name}) "
                    "SET e.type = $type, "
                    "    e.mentions = $mentions, "
                    "    e.context = $context "
                    "RETURN e"
                )
                session.run(query, 
                          name=entity["name"], 
                          type=entity["type"], 
                          mentions=entity["mentions"],
                          context=entity.get("context", ""))
    
    def create_topic_nodes(self, topics):
        """Create Topic nodes"""
        with self.driver.session() as session:
            for topic in topics:
                if not self.is_numeric(topic["word"]):
                    query = (
                        "MERGE (t:Topic {word: $word}) "
                        "SET t.frequency = $frequency, "
                        "    t.context = $context "
                        "RETURN t"
                    )
                    session.run(query, 
                             word=topic["word"], 
                             frequency=topic["frequency"],
                             context=topic.get("context", ""))
    
    def is_numeric(self, text):
        """Check if a string is numeric"""
        if not text:
            return False
        # Remove punctuation that could appear in numeric values
        text = str(text).replace(',', '').replace('.', '').replace('-', '')
        return text.isdigit() or text.replace('.', '', 1).isdigit()
    
    def create_utterance_nodes(self, utterances):
        """Create Utterance nodes"""
        with self.driver.session() as session:
            for i, utterance in enumerate(utterances):
                query = (
                    "MERGE (u:Utterance {id: $id}) "
                    "SET u.text = $text, "
                    "    u.time = $time "
                    "WITH u "
                    "MATCH (s:Speaker {name: $speaker}) "
                    "MERGE (s)-[:SPOKE]->(u) "
                    "RETURN u"
                )
                session.run(query, 
                          id=i, 
                          text=utterance["text"], 
                          time=utterance["time"],
                          speaker=utterance["speaker"])
    
    def create_relationship_edges(self, relationships):
        """Create relationship edges"""
        with self.driver.session() as session:
            for i, rel in enumerate(relationships):
                # Skip numeric values
                if (self.is_numeric(rel["subject"]) or 
                    self.is_numeric(rel["predicate"]) or 
                    self.is_numeric(rel["object"])):
                    continue
                
                try:
                    query = (
                        "MERGE (s:Entity {name: $subject}) "
                        "MERGE (o:Entity {name: $object}) "
                        "MERGE (s)-[r:RELATION {predicate: $predicate}]->(o) "
                        "SET r.context = $context, "
                        "    r.confidence = $confidence, "
                        "    r.speaker = $speaker "
                        "RETURN r"
                    )
                    session.run(query, 
                             subject=rel["subject"], 
                             object=rel["object"], 
                             predicate=rel["predicate"],
                             context=rel.get("context", ""),
                             confidence=rel["confidence"],
                             speaker=rel["speaker"])
                except Exception as e:
                    print(f"Error creating relationship: {e}")
    
    def create_activity_nodes(self, activities):
        """Create Activity nodes"""
        with self.driver.session() as session:
            for i, activity in enumerate(activities):
                query = (
                    "CREATE (a:Activity {id: $id, name: $name, context: $context, speaker: $speaker}) "
                    "RETURN a"
                )
                session.run(query, 
                          id=i, 
                          name=activity["activity"],
                          context=activity["context"],
                          speaker=activity["speaker"])
    
    def create_temporal_marker_nodes(self, markers):
        """Create TemporalMarker nodes"""
        with self.driver.session() as session:
            for i, marker in enumerate(markers):
                query = (
                    "CREATE (t:TemporalMarker {id: $id, marker: $marker, text: $text, speaker: $speaker}) "
                    "RETURN t"
                )
                session.run(query, 
                          id=i, 
                          marker=marker["marker"],
                          text=marker["text"],
                          speaker=marker["speaker"])
    
    def create_speaker_topic_edges(self, utterances, topics):
        """Connect speakers to topics"""
        with self.driver.session() as session:
            # For each speaker, find topics in their utterances
            for topic in topics:
                # Skip numeric topics
                if self.is_numeric(topic["word"]):
                    continue
                
                for speaker, _ in set((u["speaker"], 1) for u in utterances):
                    # Count how many of this speaker's utterances mention this topic
                    mentions = 0
                    for utterance in utterances:
                        if (utterance["speaker"] == speaker and 
                            topic["word"] in utterance["text"].lower()):
                            mentions += 1
                    
                    if mentions > 0:
                        query = (
                            "MATCH (s:Speaker {name: $speaker}) "
                            "MATCH (t:Topic {word: $topic}) "
                            "MERGE (s)-[r:DISCUSSES {mentions: $mentions}]->(t) "
                            "RETURN r"
                        )
                        try:
                            session.run(query, 
                                     speaker=speaker, 
                                     topic=topic["word"], 
                                     mentions=mentions)
                        except Exception as e:
                            print(f"Error connecting speaker to topic: {e}")
    
    def import_knowledge_graph(self, knowledge_graph):
        """Import the complete knowledge graph to Neo4j"""
        try:
            print("\nCreating nodes...")
            print("Creating speaker nodes...")
            self.create_speaker_nodes(knowledge_graph["speakers"])
            
            print("Creating entity nodes...")
            self.create_entity_nodes(knowledge_graph["entities"])
            
            print("Creating topic nodes...")
            self.create_topic_nodes(knowledge_graph["topics"])
            
            print("Creating utterance nodes...")
            self.create_utterance_nodes(knowledge_graph["utterances"])
            
            print("Creating activity nodes...")
            self.create_activity_nodes(knowledge_graph["activities"])
            
            print("Creating temporal marker nodes...")
            self.create_temporal_marker_nodes(knowledge_graph["temporal_markers"])
            
            print("\nCreating edges...")
            print("Creating relationship edges...")
            self.create_relationship_edges(knowledge_graph["relationships"])
            
            print("Creating speaker-topic connections...")
            self.create_speaker_topic_edges(knowledge_graph["utterances"], knowledge_graph["topics"])
            
            print("\nKnowledge graph successfully imported to Neo4j")
        except Exception as e:
            print(f"Error importing knowledge graph: {e}")
            import traceback
            traceback.print_exc()


def main():
    try:
        # Neo4j connection details
        NEO4J_URI = "neo4j+s://3d9ef5e8.databases.neo4j.io"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "sUGXID3Js-qN3aq9_SlHKMpjXJjvYuFfDziQZIGbGUs"
        
        print(f"Using Neo4j connection details: {NEO4J_URI}, {NEO4J_USERNAME}")
        
        # Directory path
        base_dir = r"C:\Users\me\OneDrive\Documents\Noyce\KnowledgeGraphs"
        
        # File path to transcript
        file_path = os.path.join(base_dir, "depaul1b.txt")
        print(f"Looking for transcript file at: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: Transcript file not found at {file_path}")
            return
        else:
            print("Transcript file found!")
        
        # Parse the transcript and build knowledge graph
        print("Starting transcript parsing...")
        parser = TranscriptParser(file_path)
        knowledge_graph = parser.build_knowledge_graph()
        
        # Connect to Neo4j and import the knowledge graph
        print("Connecting to Neo4j...")
        connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        print("Importing knowledge graph to Neo4j...")
        connector.import_knowledge_graph(knowledge_graph)
        
        print("Closing Neo4j connection...")
        connector.close()
        
        print("Process completed successfully!")
    except Exception as e:
        print(f"An error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
