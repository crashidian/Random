import os
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Set the API key directly
API_KEY = "API KEY HERE!!!! DO NOT FORGET IT!!!!!"

# Try to import required libraries, with helpful error messages if missing
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found. Please install it with 'pip install openai'")
    exit(1)

try:
    import spacy
    # Load spaCy language model for NLP processing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("spaCy package not found. Please install it with 'pip install spacy'")
    exit(1)

#####GET RID OF HARD CODINGS AND USE WORDNET!!!!!!!!!!!!!!!!!!!!!!!!!!!!#############
#TENSE AUXILARIES AND #VERB CONJUGATIONS###
class TemporalLogicSolver:
    """
    Implementation of Natural Language Temporal Logic Solver (NLTLS).
    This class transforms natural language temporal expressions into formal logic.
    """
    
    def __init__(self):
        # Temporal operators and relations
        self.temporal_relations = {
            "before": lambda t1, t2: t1 < t2,
            "after": lambda t1, t2: t1 > t2,
            "during": lambda t1, t2, t3: t1 <= t2 <= t3,
            "by": lambda t1, t2: t1 <= t2,
            "until": lambda t1, t2: t1 <= t2,
            "since": lambda t1, t2: t1 >= t2
        }
        
        # Temporal reference terms
        self.temporal_terms = [
            "before", "after", "during", "by", "until", "since", 
            "earlier", "later", "now", "today", "tomorrow", "yesterday",
            "morning", "afternoon", "evening", "night",
            "next", "last", "previous", "following", "upcoming"
        ]
        
        # Standard activity types
        self.activity_types = [
            "meeting", "call", "email", "presentation", "work", "study",
            "breakfast", "lunch", "dinner", "coffee", "exercise", "sleep",
            "go", "visit", "pick", "drop", "finish", "start", "submit",
            "review", "attend", "complete", "write", "read", "wait"
        ]
        
        # Time of day references
        self.time_of_day = {
            "morning": {"start": "06:00", "end": "12:00"},
            "noon": {"start": "12:00", "end": "12:30"},
            "afternoon": {"start": "12:00", "end": "17:00"},
            "evening": {"start": "17:00", "end": "21:00"},
            "night": {"start": "21:00", "end": "06:00"}
        }
    
    def extract_temporal_references(self, doc) -> List[Dict[str, Any]]:
        """
        Extract temporal references from the spaCy-processed text.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            List of temporal references
        """
        temporal_refs = []
        
        # 1. Extract specific time mentions (e.g. "3:00 PM")
        time_pattern = r"(\d{1,2}):(\d{2})(?:\s*(am|pm|AM|PM))?"
        for match in re.finditer(time_pattern, doc.text):
            hour = int(match.group(1))
            minute = int(match.group(2))
            am_pm = match.group(3).lower() if match.group(3) else None
            
            # Convert to 24-hour format if AM/PM is specified
            if am_pm:
                if am_pm == "pm" and hour < 12:
                    hour += 12
                elif am_pm == "am" and hour == 12:
                    hour = 0
            
            # Validate time values
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                time_str = f"{hour:02d}:{minute:02d}"
                
                # Get context around the match
                start = max(0, match.start() - 40)
                end = min(len(doc.text), match.end() + 40)
                context = doc.text[start:end]
                
                temporal_refs.append({
                    "type": "specific_time",
                    "value": time_str,
                    "raw_text": match.group(0),
                    "context": context,
                    "span": (match.start(), match.end())
                })
        
        # 2. Extract time-of-day references
        for tod, time_range in self.time_of_day.items():
            pattern = rf"\b{tod}\b"
            for match in re.finditer(pattern, doc.text.lower()):
                start = max(0, match.start() - 40)
                end = min(len(doc.text), match.end() + 40)
                context = doc.text[start:end]
                
                temporal_refs.append({
                    "type": "time_of_day",
                    "value": tod,
                    "start_time": time_range["start"],
                    "end_time": time_range["end"],
                    "raw_text": match.group(0),
                    "context": context,
                    "span": (match.start(), match.end())
                })
        
        # 3. Extract temporal relation terms
        for term in self.temporal_terms:
            pattern = rf"\b{term}\b"
            for match in re.finditer(pattern, doc.text.lower()):
                # Skip if already counted as part of a time-of-day reference
                if term in self.time_of_day:
                    continue
                    
                start = max(0, match.start() - 40)
                end = min(len(doc.text), match.end() + 40)
                context = doc.text[start:end]
                
                temporal_refs.append({
                    "type": "temporal_relation",
                    "value": term,
                    "raw_text": match.group(0),
                    "context": context,
                    "span": (match.start(), match.end())
                })
        
        # 4. Look for dates and times from spaCy's entity recognition
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                # Check if already covered by the regex patterns above
                overlap = False
                for ref in temporal_refs:
                    if (ref["span"][0] <= ent.start_char and 
                        ref["span"][1] >= ent.end_char):
                        overlap = True
                        break
                
                if not overlap:
                    temporal_refs.append({
                        "type": "entity_time",
                        "value": ent.text,
                        "raw_text": ent.text,
                        "context": ent.sent.text if hasattr(ent, 'sent') else doc.text,
                        "span": (ent.start_char, ent.end_char)
                    })
        
        # Sort references by their position in the text
        temporal_refs.sort(key=lambda x: x["span"][0])
        
        return temporal_refs
    
    def extract_activities(self, doc) -> List[Dict[str, Any]]:
        """
        Extract activities from the text.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            List of activities
        """
        activities = []
        
        # 1. Look for verb phrases (activities often involve verbs)
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in self.activity_types:
                # Get full verb phrase
                verb_phrase = ""
                
                # Include the verb itself
                verb_phrase = token.text
                
                # Find direct objects
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        # Get the entire noun phrase if possible
                        for chunk in doc.noun_chunks:
                            if child.i >= chunk.start and child.i < chunk.end:
                                verb_phrase = f"{verb_phrase} {chunk.text}"
                                break
                        else:
                            verb_phrase = f"{verb_phrase} {child.text}"
                
                # Only add if we have more than just the verb
                if len(verb_phrase.split()) > 1:
                    activities.append({
                        "type": "verb_phrase",
                        "value": verb_phrase,
                        "lemma": token.lemma_,
                        "raw_text": verb_phrase,
                        "context": token.sent.text if hasattr(token, 'sent') else doc.text,
                        "span": (token.idx, token.idx + len(verb_phrase))
                    })
        
        # 2. Look for noun phrases that match common activities
        for chunk in doc.noun_chunks:
            for word in chunk:
                if word.lemma_ in self.activity_types or word.text.lower() in self.activity_types:
                    activities.append({
                        "type": "noun_phrase",
                        "value": chunk.text,
                        "lemma": word.lemma_,
                        "raw_text": chunk.text,
                        "context": chunk.sent.text if hasattr(chunk, 'sent') else doc.text,
                        "span": (chunk.start_char, chunk.end_char)
                    })
                    break
        
        # 3. Look for entities that might represent activities
        for ent in doc.ents:
            if ent.label_ == "EVENT":
                activities.append({
                    "type": "named_entity",
                    "value": ent.text,
                    "entity_type": ent.label_,
                    "raw_text": ent.text,
                    "context": ent.sent.text if hasattr(ent, 'sent') else doc.text,
                    "span": (ent.start_char, ent.end_char)
                })
        
        # Remove duplicates and sort
        unique_activities = []
        seen_values = set()
        
        for activity in sorted(activities, key=lambda x: x["span"][0]):
            norm_value = activity["value"].lower()
            if norm_value not in seen_values:
                seen_values.add(norm_value)
                unique_activities.append(activity)
        
        return unique_activities
    
    def identify_temporal_dependencies(self, 
                                      doc,
                                      activities: List[Dict[str, Any]], 
                                      temporal_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify temporal dependencies between activities.
        
        Args:
            doc: spaCy processed document
            activities: Extracted activities
            temporal_refs: Extracted temporal references
            
        Returns:
            List of temporal dependencies
        """
        dependencies = []
        
        # Identify activities related by temporal indicators
        for temporal_ref in temporal_refs:
            if temporal_ref["type"] == "temporal_relation":
                indicator = temporal_ref["value"]
                span = temporal_ref["span"]
                
                # Find activities before and after this temporal indicator
                before_activities = [a for a in activities if a["span"][1] < span[0]]
                after_activities = [a for a in activities if a["span"][0] > span[1]]
                
                # Sort by proximity to the temporal reference
                before_activities.sort(key=lambda a: span[0] - a["span"][1])
                after_activities.sort(key=lambda a: a["span"][0] - span[1])
                
                # Create dependencies based on the indicator type
                if indicator in ["before", "prior", "earlier"]:
                    if before_activities and after_activities:
                        dependencies.append({
                            "from_activity": before_activities[0]["value"],
                            "to_activity": after_activities[0]["value"],
                            "relation": "before",
                            "indicator": indicator,
                            "confidence": 0.9
                        })
                elif indicator in ["after", "following", "later"]:
                    if before_activities and after_activities:
                        dependencies.append({
                            "from_activity": after_activities[0]["value"],
                            "to_activity": before_activities[0]["value"],
                            "relation": "before",
                            "indicator": indicator,
                            "confidence": 0.9
                        })
                elif indicator in ["by", "until"]:
                    # Look for time references
                    for time_ref in temporal_refs:
                        if time_ref["type"] in ["specific_time", "time_of_day"]:
                            if abs(time_ref["span"][0] - span[1]) < 20 or abs(time_ref["span"][1] - span[0]) < 20:
                                # Find closest activity before the indicator
                                if before_activities:
                                    dependencies.append({
                                        "from_activity": before_activities[0]["value"],
                                        "relation": "before",
                                        "absolute_time": time_ref.get("value", time_ref.get("start_time")),
                                        "indicator": indicator,
                                        "confidence": 0.8
                                    })
        
        # Detect implicit sequence from order of mention
        # If activities appear in a single sentence without explicit temporal markers,
        # their order might imply sequence (with lower confidence)
        for sent in doc.sents:
            sent_activities = [a for a in activities if sent.start_char <= a["span"][0] and sent.end_char >= a["span"][1]]
            
            if len(sent_activities) >= 2:
                # Sort by position in sentence
                sent_activities.sort(key=lambda a: a["span"][0])
                
                # Check for "and" or similar conjunctions
                for i in range(len(sent_activities) - 1):
                    activity1 = sent_activities[i]
                    activity2 = sent_activities[i + 1]
                    
                    # Check if already have a dependency
                    has_dependency = False
                    for dep in dependencies:
                        if ((dep.get("from_activity") == activity1["value"] and dep.get("to_activity") == activity2["value"]) or
                            (dep.get("from_activity") == activity2["value"] and dep.get("to_activity") == activity1["value"])):
                            has_dependency = True
                            break
                    
                    if not has_dependency:
                        # Check text between activities for conjunctions
                        between_text = doc.text[activity1["span"][1]:activity2["span"][0]].lower()
                        if "and" in between_text or "then" in between_text:
                            dependencies.append({
                                "from_activity": activity1["value"],
                                "to_activity": activity2["value"],
                                "relation": "before",
                                "indicator": "sequential_mention",
                                "confidence": 0.6
                            })
        
        # Remove duplicates, keeping highest confidence
        filtered_deps = {}
        for dep in dependencies:
            if "to_activity" in dep:
                key = f"{dep['from_activity']}_{dep['relation']}_{dep['to_activity']}"
            else:
                key = f"{dep['from_activity']}_{dep['relation']}_{dep.get('absolute_time', '')}"
            
            if key not in filtered_deps or filtered_deps[key]['confidence'] < dep['confidence']:
                filtered_deps[key] = dep
        
        return list(filtered_deps.values())
    
    def check_temporal_consistency(self, dependencies: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Check if temporal dependencies are consistent (no cycles).
        
        Args:
            dependencies: List of temporal dependencies
            
        Returns:
            Tuple of (is_consistent, explanation)
        """
        # Build a directed graph from dependencies
        graph = {}
        
        for dep in dependencies:
            from_activity = dep.get("from_activity")
            to_activity = dep.get("to_activity")
            
            if from_activity and from_activity not in graph:
                graph[from_activity] = []
            
            if to_activity and to_activity not in graph:
                graph[to_activity] = []
            
            if from_activity and to_activity:
                graph[from_activity].append(to_activity)
        
        # Check for cycles using DFS
        def is_cyclic(graph):
            visited = set()
            rec_stack = set()
            
            def is_cyclic_util(vertex):
                visited.add(vertex)
                rec_stack.add(vertex)
                
                for neighbor in graph.get(vertex, []):
                    if neighbor not in visited:
                        if is_cyclic_util(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                
                rec_stack.remove(vertex)
                return False
            
            for vertex in graph:
                if vertex not in visited:
                    if is_cyclic_util(vertex):
                        return True
            return False
        
        if is_cyclic(graph):
            return False, "Temporal constraints are inconsistent: cycle detected."
        
        return True, "Temporal constraints are consistent."
    
    def convert_to_formal_logic(self, dependencies: List[Dict[str, Any]]) -> str:
        """
        Convert temporal dependencies to formal temporal logic.
        
        Args:
            dependencies: List of temporal dependencies
            
        Returns:
            String representation of formal temporal logic formula
        """
        logic_statements = []
        
        # Process activity-to-activity dependencies
        for dep in dependencies:
            if "to_activity" in dep and dep["relation"] == "before":
                logic_statements.append(f"Occurs({dep['from_activity']}) < Occurs({dep['to_activity']})")
        
        # Process absolute time constraints
        for dep in dependencies:
            if "absolute_time" in dep and dep["absolute_time"]:
                activity = dep["from_activity"]
                relation = dep["relation"]
                time = dep["absolute_time"]
                
                if relation == "before":
                    logic_statements.append(f"Occurs({activity}) < Time({time})")
                elif relation == "after":
                    logic_statements.append(f"Occurs({activity}) > Time({time})")
                elif relation == "by":
                    logic_statements.append(f"Occurs({activity}) ≤ Time({time})")
        
        # Apply MTL (Metric Temporal Logic) formalization
        if logic_statements:
            mtl_formula = " ∧ ".join(logic_statements)
            return f"MTL: {mtl_formula}"
        else:
            return "No formal temporal logic constraints identified."
    
    def topological_sort(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """
        Sort activities based on temporal dependencies.
        
        Args:
            dependencies: List of temporal dependencies
            
        Returns:
            List of activities in temporally consistent order
        """
        # Extract all activities from dependencies
        activities = set()
        graph = {}
        
        for dep in dependencies:
            from_activity = dep.get("from_activity")
            to_activity = dep.get("to_activity")
            
            if from_activity:
                activities.add(from_activity)
                if from_activity not in graph:
                    graph[from_activity] = []
            
            if to_activity:
                activities.add(to_activity)
                if to_activity not in graph:
                    graph[to_activity] = []
            
            if from_activity and to_activity and dep["relation"] == "before":
                graph[from_activity].append(to_activity)
        
        # Perform topological sort
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            """Recursive DFS helper function for topological sort"""
            if node in temp_visited:
                # Cycle detected, skip this node
                return False
            if node in visited:
                return True
            
            temp_visited.add(node)
            
            # Visit all dependencies first
            for neighbor in graph.get(node, []):
                if not visit(neighbor):
                    # If we can't visit a neighbor, we have a cycle
                    # In this case, we'll just continue and produce a best-effort ordering
                    pass
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
            return True
        
        # Try to visit all nodes
        for node in sorted(graph.keys()):
            if node not in visited:
                visit(node)
        
        # Since we're adding nodes after visiting all dependencies,
        # we need to reverse the list to get the correct order
        result.reverse()
        
        # Add any activities not in the result (no dependencies)
        for activity in activities:
            if activity not in result:
                result.append(activity)
        
        return result
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query with temporal expressions.
        
        Args:
            query: Natural language query
            
        Returns:
            Complete analysis of temporal information
        """
        # Process text with spaCy
        doc = nlp(query)
        
        # Extract temporal references and activities
        temporal_refs = self.extract_temporal_references(doc)
        activities = self.extract_activities(doc)
        
        # Identify temporal dependencies
        dependencies = self.identify_temporal_dependencies(doc, activities, temporal_refs)
        
        # Check temporal consistency
        is_consistent, explanation = self.check_temporal_consistency(dependencies)
        
        # Generate formal logic representation
        formal_logic = self.convert_to_formal_logic(dependencies)
        
        # Create temporal sequence if consistent
        if is_consistent:
            ordered_activities = self.topological_sort(dependencies)
            sequence_status = "success"
            sequence_message = "Temporal sequence created successfully"
        else:
            ordered_activities = []
            sequence_status = "error"
            sequence_message = "Cannot create sequence due to inconsistent constraints"
        
        # Prepare the response
        response = {
            "original_query": query,
            "extracted_information": {
                "activities": [act["value"] for act in activities],
                "temporal_references": temporal_refs,
                "dependencies": dependencies
            },
            "temporal_validation": {
                "is_valid": is_consistent,
                "explanation": explanation,
                "formal_logic": formal_logic
            },
            "temporal_sequence": {
                "status": sequence_status,
                "message": sequence_message,
                "sequence": ordered_activities
            }
        }
        
        return response


class TemporalReasoningSystem:
    """
    Main system that combines rule-based processing with LLM enhancement.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the system with API key.
        
        Args:
            api_key: OpenAI API key (optional)
        """
        # Initialize OpenAI client
        self.api_key = api_key or API_KEY
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize temporal logic solver
        self.solver = TemporalLogicSolver()
    
    def llm_enhance(self, query: str, rule_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to enhance rule-based results.
        
        Args:
            query: Original query
            rule_results: Results from rule-based processing
            
        Returns:
            Enhanced results
        """
        try:
            # Format input for the LLM
            prompt = f"""
            As a temporal reasoning expert, analyze this query and the initial analysis:
            
            QUERY: {query}
            
            INITIAL ANALYSIS:
            Activities: {', '.join(rule_results['extracted_information']['activities'])}
            Dependencies: {json.dumps(rule_results['extracted_information']['dependencies'], indent=2)}
            
            Please enhance this analysis by:
            1. Identifying any missed activities or temporal dependencies
            2. Improving confidence scores if appropriate
            3. Resolving any ambiguities in temporal expressions
            
            Respond with a JSON in this format:
            {{
                "additional_activities": ["activity1", "activity2"],
                "enhanced_dependencies": [
                    {{"from_activity": "activity1", "to_activity": "activity2", "relation": "before", "confidence": 0.9}}
                ],
                "notes": "Any additional observations about temporal logic"
            }}
            """
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a temporal logic expert that can identify temporal relationships in natural language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Parse the response
            enhancement = {}
            try:
                content = response.choices[0].message.content
                # Extract JSON from the response
                json_match = re.search(r'({.+})', content.replace('\n', ' '), re.DOTALL)
                if json_match:
                    enhancement = json.loads(json_match.group(1))
                else:
                    enhancement = json.loads(content)
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                enhancement = {
                    "additional_activities": [],
                    "enhanced_dependencies": [],
                    "notes": "Error parsing LLM response"
                }
            
            return enhancement
            
        except Exception as e:
            print(f"Error in LLM enhancement: {e}")
            return {
                "additional_activities": [],
                "enhanced_dependencies": [],
                "notes": f"Error: {str(e)}"
            }
    
    def process_request(self, query: str) -> Dict[str, Any]:
        """
        Process a request with both rule-based and LLM-enhanced approaches.
        
        Args:
            query: Natural language request
            
        Returns:
            Complete analysis
        """
        # Step 1: Process with rule-based approach
        rule_results = self.solver.process_query(query)
        
        # Step 2: Enhance with LLM (if available)
        try:
            enhancements = self.llm_enhance(query, rule_results)
            
            # Step 3: Merge the results
            # Add additional activities
            rule_activities = set(rule_results["extracted_information"]["activities"])
            for activity in enhancements.get("additional_activities", []):
                if activity not in rule_activities:
                    rule_results["extracted_information"]["activities"].append(activity)
            
            # Add enhanced dependencies
            rule_dependencies = rule_results["extracted_information"]["dependencies"]
            for enhanced_dep in enhancements.get("enhanced_dependencies", []):
                # Check if this dependency already exists
                exists = False
                for rule_dep in rule_dependencies:
                    if (rule_dep.get("from_activity") == enhanced_dep.get("from_activity") and
                        rule_dep.get("to_activity") == enhanced_dep.get("to_activity") and
                        rule_dep.get("relation") == enhanced_dep.get("relation")):
                        # Update confidence if enhanced confidence is higher
                        if enhanced_dep.get("confidence", 0) > rule_dep.get("confidence", 0):
                            rule_dep["confidence"] = enhanced_dep["confidence"]
                        exists = True
                        break
                
                # Add if it doesn't exist
                if not exists:
                    rule_dependencies.append(enhanced_dep)
            
            # Re-check consistency with enhanced dependencies
            is_consistent, explanation = self.solver.check_temporal_consistency(rule_dependencies)
            rule_results["temporal_validation"]["is_valid"] = is_consistent
            rule_results["temporal_validation"]["explanation"] = explanation
            
            # Update formal logic
            rule_results["temporal_validation"]["formal_logic"] = self.solver.convert_to_formal_logic(rule_dependencies)
            
            # Update temporal sequence
            if is_consistent:
                ordered_activities = self.solver.topological_sort(rule_dependencies)
                rule_results["temporal_sequence"]["status"] = "success"
                rule_results["temporal_sequence"]["message"] = "Temporal sequence created successfully"
                rule_results["temporal_sequence"]["sequence"] = ordered_activities
            else:
                rule_results["temporal_sequence"]["status"] = "error"
                rule_results["temporal_sequence"]["message"] = "Cannot create sequence due to inconsistent constraints"
                rule_results["temporal_sequence"]["sequence"] = []
            
            # Add enhancement notes
            rule_results["llm_enhancement"] = {
                "notes": enhancements.get("notes", "No additional notes")
            }
            
        except Exception as e:
            print(f"Error combining results: {e}")
            # Add error note
            rule_results["llm_enhancement"] = {
                "notes": f"Error in LLM enhancement: {str(e)}"
            }
        
        return rule_results


# Example usage
def main():
    """
    Main function demonstrating temporal reasoning system.
    """
    # Initialize the system
    system = TemporalReasoningSystem()
    
    # Example queries
    example_queries = [
        "I need to finish my report by 5 PM, but first I need to attend a meeting at 1 PM.",
        "After I drop the kids at school in the morning, I'll go to the gym and then to work.",
        "I need to submit my assignment by midnight. Before that, I need to review it for about 30 minutes.",
        "Tomorrow, I want to go for a run, then have breakfast, and finally head to work by 9 AM.",
        "I need to pick up groceries after work, and then cook dinner before 8 PM."
    ]
    
    # Process example queries
    for i, query in enumerate(example_queries):
        print(f"\n{'='*80}\nExample {i+1}: {query}\n{'='*80}")
        
        start_time = time.time()
        result = system.process_request(query)
        end_time = time.time()
        
        # Print key results
        print("\nExtracted Activities:")
        print(", ".join(result["extracted_information"]["activities"]))
        
        print("\nTemporal Dependencies:")
        for dep in result["extracted_information"]["dependencies"]:
            if "to_activity" in dep:
                print(f"- {dep['from_activity']} {dep['relation']} {dep['to_activity']} (confidence: {dep.get('confidence', 'N/A')})")
            elif "absolute_time" in dep:
                print(f"- {dep['from_activity']} {dep['relation']} {dep['absolute_time']} (confidence: {dep.get('confidence', 'N/A')})")
        
        print("\nTemporal Validation:")
        print(f"Valid: {result['temporal_validation']['is_valid']}")
        print(f"Explanation: {result['temporal_validation']['explanation']}")
        print(f"Formal Logic: {result['temporal_validation']['formal_logic']}")
        
        print("\nTemporal Sequence:")
        if result['temporal_sequence']['status'] == 'success':
            print("Temporally consistent sequence:")
            for i, activity in enumerate(result['temporal_sequence']['sequence']):
                print(f"{i+1}. {activity}")
        else:
            print(f"Sequence generation issue: {result['temporal_sequence']['message']}")
        
        print(f"\nProcessing Time: {end_time - start_time:.2f} seconds")
    
    # Interactive mode
    print("\n" + "="*80)
    print("Interactive Mode - Enter your requests with temporal constraints (type 'quit' to exit)")
    print("="*80)
    
    while True:
        try:
            print("\nEnter your request:")
            query = input("> ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            # Process the query
            start_time = time.time()
            result = system.process_request(query)
            end_time = time.time()
            
            # Print key results
            print("\nExtracted Activities:")
            print(", ".join(result["extracted_information"]["activities"]))
            
            print("\nTemporal Dependencies:")
            for dep in result["extracted_information"]["dependencies"]:
                if "to_activity" in dep:
                    print(f"- {dep['from_activity']} {dep['relation']} {dep['to_activity']} (confidence: {dep.get('confidence', 'N/A')})")
                elif "absolute_time" in dep:
                    print(f"- {dep['from_activity']} {dep['relation']} {dep['absolute_time']} (confidence: {dep.get('confidence', 'N/A')})")
            
            print("\nTemporal Validation:")
            print(f"Valid: {result['temporal_validation']['is_valid']}")
            print(f"Explanation: {result['temporal_validation']['explanation']}")
            print(f"Formal Logic: {result['temporal_validation']['formal_logic']}")
            
            print("\nTemporal Sequence:")
            if result['temporal_sequence']['status'] == 'success':
                print("Temporally consistent sequence:")
                for i, activity in enumerate(result['temporal_sequence']['sequence']):
                    print(f"{i+1}. {activity}")
            else:
                print(f"Sequence generation issue: {result['temporal_sequence']['message']}")
            
            print(f"\nProcessing Time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing request: {e}")


if __name__ == "__main__":
    main()
