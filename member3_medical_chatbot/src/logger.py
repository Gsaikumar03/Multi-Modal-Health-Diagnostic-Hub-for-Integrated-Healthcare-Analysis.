import logging

logging.basicConfig(
    filename="rag_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_interaction(question, answer):
    logging.info(f"Q: {question}")
    logging.info(f"A: {answer}")