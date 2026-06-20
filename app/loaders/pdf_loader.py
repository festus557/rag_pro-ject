from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys

class  PdfLoader:
    def __init__(self, dir_path, chunk_size = 400, chunk_overlap=80):
        self.dir_path = dir_path
        self.all_files = []
        self.all_pages = []
        self.chunks = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

    def get_files(self):
        if not os.path.exists(self.dir_path):
              sys.exit("Directory Not Found")

        files = os.listdir(self.dir_path)
        self.all_files.extend(files)

    def get_pages(self):
             if len(self.all_files) > 0 :
                  for file in self.all_files:
                       full_path = self.dir_path + "/" + file
                       loader = PyPDFLoader(full_path)
                       pages = loader.load()
                       self.all_pages.extend(pages)

    def get_splitted(self):
         if len(self.all_pages) > 0:
               splitter = RecursiveCharacterTextSplitter(
                    chunk_size = self.chunk_size,
                    chunk_overlap = self.chunk_overlap
               )

               chunks = splitter.split_documents(self.all_pages)
               self.chunks = chunks

    def run(self):
         try:
              self.get_files()
              self.get_pages()
              self.get_splitted()
         except Exception as e:
              print(f"[+]Error: {str(e)}")

    def get_chunks(self):
         self.run()
         return self.chunks

if __name__ == "__main__":
         path = "/home/festus/projects/AI_projects/test_rag/data"
         pdfloader = PdfLoader(dir_path=path)

         chunks = pdfloader.get_chunks()
         print(len(chunks))