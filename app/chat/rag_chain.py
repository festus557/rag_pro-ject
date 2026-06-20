from langchain_ollama import ChatOllama
import sys

class ChatModel:
    def __init__(self, retriever, model = "qwen2.5:1.5b"):
        self.context = ""
        self.model = model
        self.retriever = retriever
        self.llm = ChatOllama(model=self.model)

    def ask_qn(self, qn):
           """
            2. If the answer is not present in the context, say 
                              "I could not find that information in the provided documents" 
                               then use outside knownledge  to get that information.
                           3. Do not use outside knownledge unless as instructed on instruction number 2.
                           4. Keep answers factual and concise.
            """
           if self.retriever is not None:
                self.context = ""
                retrieved_chunk = self.retriever.invoke(qn)
                for chunk in retrieved_chunk:
                    self.context += chunk.page_content + "\n\n"
                prompt = f"""
                You are customizable ai assistant called ThinkBig created by pharmacist students from ThinkBig Family
                
                RULES
                1. Do not use any external knownledge
                2. If answer not found from the given context say "Not Found"
                Context: {self.context}
                Question: {qn}
                """
                response = self.llm.invoke(prompt)
                return  response.content
           else:
                sys.exit(f"[+]Missing arguments")