import sys
import os

sys.path.append(os.path.abspath(".."))

from member3_medical_chatbot.src.rag_pipeline import MedicalRAG
from member3_medical_chatbot.src.memory import ChatMemory
from member3_medical_chatbot.src.logger import log_interaction

class ChatService:

    def __init__(self):
        self.rag = MedicalRAG()
        self.memory = ChatMemory()

    def ask_question(self, question):

        # Get conversation memory
        memory_context = self.memory.get_context()

        # Ask Hybrid RAG (Member 3)
        response = self.rag.ask(question, memory_context)

        # Store conversation
        self.memory.add(question, response["answer"])

        # Log interaction
        log_interaction(
            question,
            response["answer"],
            response["confidence"]
        )

        return {
            "question": question,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "context": response["context_used"]
        }