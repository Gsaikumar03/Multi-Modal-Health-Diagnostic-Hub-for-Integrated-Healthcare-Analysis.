class ChatMemory:
    def __init__(self):
        self.history = []

    def add(self, question, answer):
        self.history.append({"question": question, "answer": answer})

    def get_context(self):
        context = ""
        for item in self.history[-5:]:
            context += f"Q: {item['question']}\nA: {item['answer']}\n"
        return context