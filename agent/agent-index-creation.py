
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from jproperties import Properties


loader = CSVLoader('BasePlan.csv')
data = loader.load()
#print(data)

prompt_configs = Properties()
with open('prompt_configs.properties', 'rb') as read_prop:
    prompt_configs.load(read_prop) 

print("Read properties file---------")
embedding_function = HuggingFaceInstructEmbeddings(
            #model_name="hkunlp/instructor-large",
            model_name=prompt_configs.get("HuggingFace_Embedding_Model").data,
            model_kwargs={"device": "cpu"}
        )
print(embedding_function)
base_plans = []
pageContent = ""
for ind in range(len(data)):
    pageContent = pageContent + "\n|" + data[ind].page_content

docs = pageContent.split("\n|")
for doc in docs:
    base_plan = Document(page_content=doc)
    base_plans.append(base_plan)

db = Chroma.from_documents(base_plans, embedding_function,persist_directory='./agent_db/base_plan_db')
db.persist()   

loader1 = CSVLoader('Vas.csv')
data1 = loader1.load()

vas_plans = []
pageContent1 = ""
for ind in range(len(data)):
    pageContent1 = pageContent1 + "\n|" + data1[ind].page_content

docs1 = pageContent1.split("\n|")
for doc1 in docs1:
    vas_plan = Document(page_content=doc1)
    vas_plans.append(vas_plan)

db = Chroma.from_documents(vas_plans, embedding_function,persist_directory='./agent_db/vas_plan_db')
db.persist()   