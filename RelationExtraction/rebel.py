from transformers import pipeline
from uuid import uuid4
import time

class RelationExtractor:
    def __init__(self):
        self.triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')
        # We need to use the tokenizer manually since we need special tokens.
        # Function to parse the generated text and extract the triplets
    
    def tokenize_text(self, text):
        extracted_text = self.triplet_extractor.tokenizer.batch_decode([self.triplet_extractor(text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
        return extracted_text
    
    def extract_triplets(self, text, entity_info):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        triplets_linked = []
        for triplet in triplets:
            if triplet["head"] != '' and triplet["tail"] != '' and triplet["type"] != '':
                triplet["head"] = triplet["head"].strip()
                triplet["tail"] = triplet["tail"].strip()
                if triplet["head"] in entity_info:
                    entity = entity_info[triplet["head"]]
                    if not entity["id"]:
                        entity["id"] = str(uuid4())
                    head = {"name":triplet["head"], "label": entity["label"], "id":entity["id"]}
                else:
                    head = {"name":triplet["head"], "label":"","id":str(uuid4())}
                if triplet["tail"] in entity_info:
                    entity = entity_info[triplet["tail"]]
                    if not entity["id"]:
                        entity["id"] = str(uuid4())
                    tail = {"name":triplet["tail"], "label": entity["label"], "id":entity["id"]}
                else:
                    tail = {"name":triplet["tail"], "label":"","id":str(uuid4())}
                triplets_linked.append({'head': head, 'type': triplet["type"].strip(),'tail': tail})
        return triplets_linked

if __name__ == "__main__":
    relex = RelationExtractor()
    text = "Standing tall on Liberty Island in New York Harbor, the Statue of Liberty, a colossal neoclassical sculpture crafted from copper designed by French sculptor Frédéric Auguste Bartholdi and engineered by Gustave Eiffel, serves as an enduring symbol of freedom and democracy, representing the friendship between the United States and France, welcoming immigrants with its torch of enlightenment held high, adorned in a flowing robe and crowned with seven radiant spikes symbolizing the seven seas and continents, a beacon of hope and inspiration since its dedication on October 28, 1886, and inviting all who gaze upon her to embrace the ideals of liberty, democracy, and the pursuit of happiness that define the American spirit. The construction of the Ram Mandir in Ayodhya, India, symbolizes a historic moment, marking the resolution of a long-standing cultural and legal dispute, and the fulfillment of the Hindu community's aspirations, as the sacred site now stands as a testament to the nation's commitment to religious harmony, cultural heritage, and the principles of inclusivity and unity. Punta Cana is a resort town in the municipality of Higuey, in La Altagracia Province, the eastern most province of the Dominican Republic."
    t1 = time.time()
    extracted_text = relex.tokenize_text(text)
    t2 = time.time()
    extracted_triplets = relex.extract_triplets(extracted_text[0])
    t3 = time.time()
    print(extracted_triplets)
    print(f"Tokenizer time = {t2-t1}\nModel time = {t3-t2}")