from app.chat.rag_chain import ChatModel
from app.loaders.pdf_loader import PdfLoader
from app.embeddings.bge_m3 import Embed
from app.vector_store.faiss_store import FaisRun
import sys
import random

path = "/home/festus/projects/AI_projects/test_rag/data"
save_vector_path = "/home/festus/projects/AI_projects/rag_project/saved"

def colors():
       G = "\033[32m"
       R = "\033[31m"
       B = "\033[33m"
       Y = "\033[34m"
       M = "\033[35m"
       RE = "\033[0m"

       return [G, R, B, Y, M, RE]


def load_model():
    
        pdfloader = PdfLoader(dir_path=path)
        chunks = pdfloader.get_chunks()

        embedder = Embed()
        embeddings = embedder.get_embeddings()

        faisRun = FaisRun(embed=embeddings, chunks=chunks, local_data=save_vector_path)
        retriever = faisRun.get_retriver()

        model = ChatModel(retriever=retriever)
        return model

def runer(model):
        #"Why is Elrond  investigated?"
        choice = random.choice(colors())
        qn = input(f"{choice}YOU>> \033[0m")
        qn = qn.strip()
        if len(qn) > 0:
             response = model.ask_qn(qn=qn)
             response += "\r\n"
             print(response)

def main():
        model = load_model()
        try: 
                while True:
                     runer(model=model)
        except KeyboardInterrupt:
                print(f"[-]Keyboard interrupted Closing")
                sys.exit("Exiting....")
        except Exception as e:
                print(f"[-]Error: {str(e)}")

if __name__ == "__main__":
        main()