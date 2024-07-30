import PyPDF2
import spacy
from neo4j import GraphDatabase



# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Process the text
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Step 3: Load data into Neo4j
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_entity(self, entity_name, entity_type):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_entity, entity_name, entity_type)
    
    @staticmethod
    def _create_and_return_entity(tx, entity_name, entity_type):
        query = (
            "MERGE (e:Entity {name: $entity_name, type: $entity_type}) "
            "RETURN e"
        )
        tx.run(query, entity_name=entity_name, entity_type=entity_type)

# Main script
pdf_path = "Knowledge Base-copy-updated_weekly.pdf"  # Replace with the path to your PDF file
text = extract_text_from_pdf(pdf_path)
entities = extract_entities(text)

# Replace these values with your Neo4j credentials
NEO4J_URI="neo4j+s://5b98d0eb.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="9PFwV0Q03VNoIXiIqLkGv3aBzqJirLQHbwFeqLIsFQI"


graph = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

for entity_name, entity_type in entities:
    graph.create_entity(entity_name, entity_type)

graph.close()
