# data_loader.py

import os

class DataLoader:
    def load_documents(self, directory):
        documents = []
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append((filename, text))
        return documents
