from langchain_ollama import OllamaEmbeddings, ChatOllama

class Embed:
    def __init__(self, model = "bge-m3"):
        self.embeddings = None
        self.model = model

    def embeded(self):
        self.embeddings = OllamaEmbeddings(model=self.model)

    def run(self):
        self.embeded()

    def get_embeddings(self):
         self.run()
         return self.embeddings
