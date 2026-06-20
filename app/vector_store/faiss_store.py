from langchain_community.vectorstores import FAISS

class FaisRun:
    def __init__(self, embed, chunks, local_data, k=5):
        self.embeddings = embed
        self.chunks = chunks
        self.vector_db = None
        self.retriever = None
        self.k = k
        self.local_data = local_data

    def make_vector(self):
        self.vector_db = FAISS.from_documents(self.chunks, self.embeddings)

    def save_local(self):
        self.vector_db.save_local(self.local_data)

    def get_retriver(self):
        self.make_vector()
        self.save_local()
        self.retriver = self.vector_db.as_retriever(search_kwargs={"k":self.k})
        return self.retriver

    
