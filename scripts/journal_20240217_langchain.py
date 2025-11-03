# %%
# Trying out [Graph Querying](https://python.langchain.com/docs/use_cases/graph/graph_networkx_qa)
# %%
import os
from time import time
# %%
from langchain.indexes import GraphIndexCreator
from langchain_community.llms import Ollama  # Doesn't seem to return any graph triples
# from langchain_openai import OpenAI   # Example used in doc - may be only OpenAI specific?
from langchain.chains import GraphQAChain

# %%
index_creator = GraphIndexCreator(llm=Ollama(model='zephyr:7b'))

# %%
data_dir = os.path.expanduser('~/Work/Data/lovecraftcorpus')
file_name = os.path.join(data_dir, 'dunwich.txt')
with open(file_name, 'r') as f:
    all_text = f.read()

# %%
# text = ".\n".join(all_text.split(".")[105:108])
text = 'It won’t look like much, but if you stop and look closely, you’ll see a “Field of dreams,” the ground on which America’s future will be built. \nThis is where Intel, the American company that helped build Silicon Valley, is going to build its $20 billion semiconductor “mega site”. \nUp to eight state-of-the-art factories in one place. 10,000 new good-paying jobs. '
# %%
print(text)
# %%
# This is pretty slow even for three sentences
start_time = time()
graph = index_creator.from_text(text)
end_time = time()
print(f'Time taken: {end_time - start_time}s')

# %%
print(graph.get_triples())