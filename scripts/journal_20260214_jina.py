# %%
# https://huggingface.co/jinaai/jina-reranker-v3
from transformers import AutoModel

model = AutoModel.from_pretrained(
    'jinaai/jina-reranker-v3',
    dtype="auto",
    trust_remote_code=True,
)
model.eval()


# %%
# query = "What are the health benefits of green tea?"
# documents = [
#     "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
#     "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
#     "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
#     "Basketball is one of the most popular sports in the United States.",
#     "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
#     "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
# ]

# query = 'forest'
query = 'dogs purr'
documents = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]


# Rerank documents
results = model.rerank(query, documents)

# Results are sorted by relevance score (highest first)
for result in results:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
    print()

