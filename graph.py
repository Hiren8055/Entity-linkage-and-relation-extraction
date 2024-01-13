"""build a entity linker to neo4j database
the data will be coming from model 
entity 
relation between entity

built a program that can take entity and relation, and put it to neo4j

create two nodes and add realtion between them"""



# create 
# (elon musk:Person{name:"elon musk"}),
# (jeff bezos:Person{name:"jeff bezos"}),

# (elon musk -[:friend]-> jeff bezos)};

from neo4j import GraphDatabase

class test_1:
    def __init__(self,uri, user,password) -> None:
        self.driver = GraphDatabase.driver(uri,auth=(user,password))
    
    def close(self):
        self.driver.close()
    
    def print_entity(self,message):
        with self.driver.session() as session:
            adding_node = session.execute_write(self.add_entity, message)
    
    @staticmethod    
    def add_entity(tx,message):
        # message = message[0]
        # print(message)
        query = "CREATE "
        x = ""
        a=""
        b = ""
        for i in message:
            # print(i)
            # print(i['head'])
            # print(i['tail'])
            # print(i['type'])
            
            x =x+"(" +str(i["head"].split()[0])+":person{name:'"+str(i["head"])+"'}),"
            x =x+ "(" +str(i["tail"].split()[0])+":person{name:'"+str(i["tail"])+"'}),"
            # linking ="("+a+")"+"-[:"+str(i["type"].split()[0])+"]->"+"("+b+"),"
            # c = "MATCH("+a+","+b+")" + linking
            # x = x + "("+a+")"+"-[:"+str(i["type"].split()[0])+"]->"+"("+b+"),"
            
            # print("("+str(i["head"].split()[0])+":person{name:"+str(i["head"])+")")
            # print("("+str(i["tail"].split()[0])+":person{name:"+str(i["tail"])+")")
        # for i in message:
        #     x = x+"("+str(i["head"].split()[0])+":{name:"+str(i["head"])+"),"
        print("x",x)
        # """match (a,b) create linking"""
        query = query + x
        query = query[:-1] + ";"
        print("query",query)
        result = tx.run(query)
        for i in message:
            a = str(i["head"].split()[0])+":person{name:'"+str(i["head"])+"'}"
            b = str(i["tail"].split()[0])+":person{name:'"+str(i["tail"])+"'}"
            linking ="("+a+")"+"-[:"+str(i["type"].split()[0])+"]->"+"("+b+"),"
            c = "MATCH("+a+","+b+")" + linking
            print("c",c)
            result = tx.run(query)
        return result
    
if __name__ == "__main__":
    fun = test_1("bolt://localhost:7687","neo4j","12345678")
    # message = [{'head': 'Punta Cana', 'type': 'located in the administrative territorial entity', 'tail': 'La Altagracia Province'}, {'head': 'Punta Cana', 'type': 'country', 'tail': 'Dominican Republic'}, {'head': 'Higuey', 'type': 'located in the administrative territorial entity', 'tail': 'La Altagracia Province'}, {'head': 'Higuey', 'type': 'country', 'tail': 'Dominican Republic'}, {'head': 'La Altagracia Province', 'type': 'country', 'tail': 'Dominican Republic'}, {'head': 'Dominican Republic', 'type': 'contains administrative territorial entity', 'tail': 'La Altagracia Province'}]
    # message = [{'head': 'Punta Cana', 'type': 'located in the administrative territorial entity', 'tail': 'La Altagracia Province'}, {'head': 'Higuey', 'type': 'country', 'tail': 'Dominican Republic'}]
    message = [{'head': 'Statue of Liberty', 'type': 'architectural style', 'tail': 'neoclassical'}, {'head': 'Statue of Liberty', 'type': 'creator', 'tail': 'Frédéric Auguste Bartholdi'}, {'head': 'Statue of Liberty', 'type': 'creator', 'tail': 'Gustave Eiffel'}, {'head': 'Statue of Liberty', 'type': 'inception', 'tail': 'October 28, 1886'}, {'head': 'Frédéric Auguste Bartholdi', 'type': 'notable work', 'tail': 'Statue of Liberty'}, {'head': 'Gustave Eiffel', 'type': 'notable work', 'tail': 'Statue of Liberty'}]
    fun.print_entity(message)
    fun.close()
