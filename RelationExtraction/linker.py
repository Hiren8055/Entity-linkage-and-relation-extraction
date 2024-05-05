import spacy

class EntityDisambiguator:
    def __init__(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe("entityLinker", last=True)
    
    def link_entites(self, text):
        print(type(text))
        doc = self.nlp(str(text))
        print(doc)
        entity_info = dict()
        for sent in doc.sents:
            for ent in sent._.linkedEntities:
                entity_info[ent.get_span().text.strip()] = {
                    "name":ent.get_span().text.strip(),
                    "label": ent.get_label(), 
                    "id": ent.get_id()
                    }
        return entity_info
    
if __name__ == "__main__":
    linker = EntityDisambiguator()
    info = linker.link_entites("Naveen Arora is the current Prime Minsiter of India.")
    print(info)