from typing import Callable
from .store import EmbeddingStore

class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """
        Initialize the agent with a vector store and a language model function.
        """
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant chunks from the Incoterms database and generate an answer.
        """
        # 1. Retrieve top-k relevant chunks from the store
        # Kết quả từ store.search là một list các dict chứa {"text": ..., "metadata": ...}
        relevant_chunks = self.store.search(question, top_k=top_k)

        if not relevant_chunks:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu Incoterms 2020."

        # 2. Build a prompt with the chunks as context
        # Kết hợp các đoạn văn bản lại thành một khối context duy nhất
        context_text = "\n---\n".join([chunk["content"] for chunk in relevant_chunks])

        # Thiết kế System Prompt/Instructions để định hướng Agent
        prompt = f"""
        Bạn là một chuyên gia về Incoterms® 2020. Hãy trả lời câu hỏi của người dùng dựa TRỰC TIẾP trên các đoạn trích dẫn từ tài liệu dưới đây.

        ---
        BỐI CẢNH TÀI LIỆU (CONTEXT):
        {context_text}
        ---

        YÊU CẦU:
        1. Nếu câu trả lời không có trong bối cảnh trên, hãy nói rằng bạn không biết, đừng tự ý bịa đặt thông tin.
        2. Trích dẫn rõ điều khoản (ví dụ: A2, B5) nếu thông tin đó có sẵn trong context.
        3. Câu trả lời cần ngắn gọn, chính xác và chuyên nghiệp.

        CÂU HỎI: {question}

        CÂU TRẢ LỜI:
        """

        # 3. Call the LLM to generate an answer
        response = self.llm_fn(prompt)
        
        return response