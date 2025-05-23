#生成向量数据库
def create_vector_db():
    import os
    import sys
    if not os.path.exists("db"):
        os.makedirs("db")
    import faiss
    import pickle
    from sentence_transformers import SentenceTransformer
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data.mmlu import merge_csv_files_in_folder
    dataset=merge_csv_files_in_folder(os.path.join(parent_dir,"data","MMLU","dev"))
    docs = []
    for item in dataset:
            text = item[0].replace(",please answer A,B,C,or D.",",")+f"answer:{item[1]}."
            docs.append(text)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embed_model.encode(docs)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    faiss.write_index(index, "db/index.faiss")
    with open("db/index.pkl", "wb") as f:
        pickle.dump(docs, f)

def load_vector_db():
    import faiss
    import pickle
    from sentence_transformers import SentenceTransformer
    class db:
        def __init__(self):
            self.index = faiss.read_index("db/index.faiss")
            with open("db/index.pkl", "rb") as f:
                self.docs = pickle.load(f)
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        def search(self, query, k=5):
            query_embedding = self.embed_model.encode([query])
            D, I = self.index.search(query_embedding, k)
            return [self.docs[i] for i in I[0]]
    return db()
