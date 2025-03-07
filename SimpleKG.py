import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk
from neo4j import GraphDatabase

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # Not 'averaged_perceptron_tagger_eng'
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Neo4j connection details
URI = "neo4j+s://3d9ef5e8.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "sUGXID3Js-qN3aq9_SlHKMpjXJjvYuFfDziQZIGbGUs"

# The story text
story_text = """He needs this sort of thing because of what it is outside and it's like apparently this person is saying no I don't want that. And so it was like just going away and then going out here and it's like oh that's what's happening and so that's wacky whatever came back in and had wetness on it and then got that thing which is nicer when you're when you're walking on that to keep from getting all wetness all over you but especially on your head."""

class KnowledgeGraphBuilder:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_time_nodes(self):
        with self.driver.session() as session:
            # Create time frame nodes
            time_frames = [
                {"id": 1, "description": "Initial Scene - Umbrella Offered and Refused"},
                {"id": 2, "description": "Going Outside"},
                {"id": 3, "description": "Realizing It's Raining"},
                {"id": 4, "description": "Returning Inside Wet"},
                {"id": 5, "description": "Taking the Umbrella"}
            ]
            
            for time_frame in time_frames:
                session.run(
                    "CREATE (t:TimeFrame {id: $id, description: $description})",
                    id=time_frame["id"],
                    description=time_frame["description"]
                )
            
            # Create temporal sequence between time frames
            for i in range(1, 5):
                session.run(
                    """
                    MATCH (t1:TimeFrame {id: $id1}), (t2:TimeFrame {id: $id2})
                    CREATE (t1)-[:NEXT]->(t2)
                    """,
                    id1=i,
                    id2=i+1
                )
            
            print("Time frame nodes created")
    
    def create_entity_nodes(self):
        with self.driver.session() as session:
            # Create entity nodes
            entities = [
                {"id": "person", "type": "Person", "description": "Main character in the story"},
                {"id": "other_person", "type": "Person", "description": "Person offering umbrella"},
                {"id": "umbrella", "type": "Object", "description": "The umbrella"},
                {"id": "rain", "type": "Weather", "description": "Rain outside"},
                {"id": "inside", "type": "Location", "description": "Inside location"},
                {"id": "outside", "type": "Location", "description": "Outside location"},
                {"id": "wetness", "type": "Condition", "description": "Being wet from rain"},
                {"id": "head", "type": "BodyPart", "description": "Head of the person"}
            ]
            
            for entity in entities:
                session.run(
                    "CREATE (e:Entity {id: $id, type: $type, description: $description})",
                    id=entity["id"],
                    type=entity["type"],
                    description=entity["description"]
                )
            
            print("Entity nodes created")
    
    def create_spatiotemporal_relations(self):
        with self.driver.session() as session:
            # TimeFrame 1: Initial Scene
            session.run("""
                MATCH (t:TimeFrame {id: 1}), (p:Entity {id: 'person'}), (o:Entity {id: 'other_person'}), 
                      (u:Entity {id: 'umbrella'}), (l:Entity {id: 'inside'})
                CREATE (p)-[:LOCATED_AT {timeframe: 1}]->(l)
                CREATE (o)-[:LOCATED_AT {timeframe: 1}]->(l)
                CREATE (u)-[:LOCATED_AT {timeframe: 1}]->(l)
                CREATE (o)-[:OFFERS {timeframe: 1}]->(u)
                CREATE (p)-[:REFUSES {timeframe: 1}]->(u)
                CREATE (t)-[:INCLUDES]->(p)
                CREATE (t)-[:INCLUDES]->(o)
                CREATE (t)-[:INCLUDES]->(u)
                CREATE (t)-[:INCLUDES]->(l)
            """)
            
            # TimeFrame 2: Going Outside
            session.run("""
                MATCH (t:TimeFrame {id: 2}), (p:Entity {id: 'person'}), 
                      (l:Entity {id: 'outside'})
                CREATE (p)-[:MOVES_TO {timeframe: 2}]->(l)
                CREATE (t)-[:INCLUDES]->(p)
                CREATE (t)-[:INCLUDES]->(l)
            """)
            
            # TimeFrame 3: Realizing It's Raining
            session.run("""
                MATCH (t:TimeFrame {id: 3}), (p:Entity {id: 'person'}), 
                      (r:Entity {id: 'rain'}), (l:Entity {id: 'outside'})
                CREATE (p)-[:LOCATED_AT {timeframe: 3}]->(l)
                CREATE (p)-[:EXPERIENCES {timeframe: 3}]->(r)
                CREATE (p)-[:REALIZES {timeframe: 3, what: 'mistake'}]->(r)
                CREATE (t)-[:INCLUDES]->(p)
                CREATE (t)-[:INCLUDES]->(r)
                CREATE (t)-[:INCLUDES]->(l)
            """)
            
            # TimeFrame 4: Returning Inside Wet
            session.run("""
                MATCH (t:TimeFrame {id: 4}), (p:Entity {id: 'person'}), 
                      (w:Entity {id: 'wetness'}), (l:Entity {id: 'inside'})
                CREATE (p)-[:MOVES_TO {timeframe: 4}]->(l)
                CREATE (p)-[:HAS_CONDITION {timeframe: 4}]->(w)
                CREATE (t)-[:INCLUDES]->(p)
                CREATE (t)-[:INCLUDES]->(w)
                CREATE (t)-[:INCLUDES]->(l)
            """)
            
            # TimeFrame 5: Taking the Umbrella
            session.run("""
                MATCH (t:TimeFrame {id: 5}), (p:Entity {id: 'person'}), 
                      (u:Entity {id: 'umbrella'}), (w:Entity {id: 'wetness'}),
                      (h:Entity {id: 'head'})
                CREATE (p)-[:TAKES {timeframe: 5}]->(u)
                CREATE (u)-[:PROTECTS_FROM {timeframe: 5}]->(w)
                CREATE (u)-[:ESPECIALLY_PROTECTS {timeframe: 5}]->(h)
                CREATE (t)-[:INCLUDES]->(p)
                CREATE (t)-[:INCLUDES]->(u)
                CREATE (t)-[:INCLUDES]->(w)
                CREATE (t)-[:INCLUDES]->(h)
            """)
            
            print("Spatiotemporal relations created")

    def analyze_text(self, text):
        try:
            tokens = word_tokenize(text)
            print("\nNLTK Text Analysis:")
            print("Tokens:", tokens[:10], "... (truncated)")
            
            try:
                tagged = pos_tag(tokens)
                print("POS Tags:", tagged[:10], "... (truncated)")
                
                # Extract potential entities and actions from POS tags
                nouns = [word for word, pos in tagged if pos.startswith('NN')]
                verbs = [word for word, pos in tagged if pos.startswith('VB')]
                
                print("Potential Entities (Nouns):", nouns)
                print("Potential Actions (Verbs):", verbs)
                
                try:
                    entities = ne_chunk(tagged)
                    return {"tokens": tokens, "tagged": tagged, "entities": entities}
                except Exception as e:
                    print(f"Warning: Named Entity Recognition failed: {e}")
                    return {"tokens": tokens, "tagged": tagged, "entities": None}
            
            except Exception as e:
                print(f"Warning: POS tagging failed: {e}")
                print("Continuing with knowledge graph creation without POS analysis...")
                return {"tokens": tokens, "tagged": None, "entities": None}
                
        except Exception as e:
            print(f"Warning: Text tokenization failed: {e}")
            print("Continuing with knowledge graph creation without text analysis...")
            return {"tokens": None, "tagged": None, "entities": None}

    def create_cypher_query_examples(self):
        print("\nUseful Cypher Queries:")
        
        print("\n1. View all TimeFrames:")
        print("MATCH (t:TimeFrame) RETURN t ORDER BY t.id")
        
        print("\n2. View all Entities:")
        print("MATCH (e:Entity) RETURN e.id, e.type, e.description")
        
        print("\n3. Get the complete story path:")
        print("""
        MATCH path = (t1:TimeFrame {id: 1})-[:NEXT*]->(t5:TimeFrame)
        RETURN path
        """)
        
        print("\n4. Get all relationships for a specific TimeFrame:")
        print("""
        MATCH (t:TimeFrame {id: 3})-[:INCLUDES]->(e)
        OPTIONAL MATCH (e)-[r]->(e2)
        WHERE r.timeframe = 3
        RETURN e.id, type(r), e2.id
        """)
        
        print("\n5. Track the person's journey through all TimeFrames:")
        print("""
        MATCH (p:Entity {id: 'person'})
        MATCH (t:TimeFrame)-[:INCLUDES]->(p)
        OPTIONAL MATCH (p)-[r]->(e)
        WHERE r.timeframe = t.id
        RETURN t.id, t.description, type(r), e.id
        ORDER BY t.id
        """)

def main():
    print("Starting the Knowledge Graph Builder...")
    try:
        # Initialize the knowledge graph builder
        graph = KnowledgeGraphBuilder(URI, USERNAME, PASSWORD)
        
        try:
            # Analyze the text using NLTK
            print("Analyzing text with NLTK...")
            analysis = graph.analyze_text(story_text)
            
            # Clear existing data
            print("Clearing existing Neo4j database...")
            graph.clear_database()
            
            # Create knowledge graph nodes and relationships
            print("Creating time frame nodes...")
            graph.create_time_nodes()
            
            print("Creating entity nodes...")
            graph.create_entity_nodes()
            
            print("Creating spatiotemporal relationships...")
            graph.create_spatiotemporal_relations()
            
            # Print example Cypher queries
            graph.create_cypher_query_examples()
            
            print("\nKnowledge graph successfully created in Neo4j!")
            print(f"Access your database at {URI}")
            
        except Exception as e:
            print(f"Error during graph creation: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("Closing Neo4j connection...")
            graph.close()
            
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Please check your Neo4j credentials and ensure the database is running.")

if __name__ == "__main__":
    main()
