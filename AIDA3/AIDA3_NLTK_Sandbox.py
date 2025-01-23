import nltk
from nltk import Tree, CFG, PCFG, ViterbiParser
from nltk.chunk import RegexpParser, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.parse import RecursiveDescentParser
from nltk.sem import logic
from collections import defaultdict

class NLTKAnalyzer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        self.initialize_grammar()
        
    def initialize_grammar(self):
        self.chunk_grammar = r"""
            NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
            VP: {<MD>?<VB.*><RB>*<NP|PP|SBAR>*}
            PP: {<IN><NP>}
            SBAR: {<IN><S>}
            S: {<NP><VP>}
        """
        self.chunker = RegexpParser(self.chunk_grammar)
        
    def analyze_syntax(self, text):
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'chunks': self.chunker.parse(pos_tags),
            'named_entities': ne_chunk(pos_tags)
        }

    def build_dependency_tree(self, pos_tags):
        def create_tree(head_idx, tokens):
            if not tokens:
                return None
            head = tokens[head_idx]
            tree = Tree(head[1], [head[0]])
            
            # Left dependencies
            if head_idx > 0:
                tree.insert(0, Tree('LEFT', [f"{t[0]}/{t[1]}" for t in tokens[:head_idx]]))
            
            # Right dependencies
            if head_idx < len(tokens) - 1:
                tree.append(Tree('RIGHT', [f"{t[0]}/{t[1]}" for t in tokens[head_idx + 1:]]))
            
            return tree
        
        # Find main verb as root
        root_idx = 0
        for i, (_, tag) in enumerate(pos_tags):
            if tag.startswith('VB'):
                root_idx = i 
                break
                
        return create_tree(root_idx, pos_tags)

    def analyze_semantics(self, text):
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        semantics = {
            'subjects': [],
            'verbs': [],
            'objects': [],
            'modifiers': [],
            'relationships': []
        }
        
        for i, (word, tag) in enumerate(pos_tags):
            if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']:
                if i > 0 and pos_tags[i-1][1] in ['IN', 'TO']:
                    semantics['objects'].append(word)
                else:
                    semantics['subjects'].append(word)
            elif tag.startswith('VB'):
                semantics['verbs'].append(word)
            elif tag in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                semantics['modifiers'].append(word)
                
        # Build basic relationships
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB') and i > 0 and i < len(pos_tags) - 1:
                subj = pos_tags[i-1][0]
                obj = pos_tags[i+1][0]
                semantics['relationships'].append(f"{subj} -{word}-> {obj}")
                
        return semantics

def main():
    analyzer = NLTKAnalyzer()
    
    while True:
        text = input("\nEnter a sentence to analyze (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
            
        print("\n=== Syntactic Analysis ===")
        syntax = analyzer.analyze_syntax(text)
        print("\nPOS Tags:")
        print(syntax['pos_tags'])
        print("\nChunks:")
        syntax['chunks'].pretty_print()
        print("\nNamed Entities:")
        syntax['named_entities'].pretty_print()
        
        print("\n=== Dependency Analysis ===")
        dep_tree = analyzer.build_dependency_tree(syntax['pos_tags'])
        dep_tree.pretty_print()
        
        print("\n=== Semantic Analysis ===")
        semantics = analyzer.analyze_semantics(text)
        for category, items in semantics.items():
            if items:
                print(f"\n{category.capitalize()}:")
                for item in items:
                    print(f"  {item}")

if __name__ == "__main__":
    main()