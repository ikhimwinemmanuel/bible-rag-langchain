import gradio as gr

from src.rag_qa import answer_question

def chat_fn(message, history):
    """
    This function is called every time the user sends a message.
    """
    answer = answer_question(message)
    return answer

demo = gr.ChatInterface(
    fn=chat_fn,
    title=" Bible RAG Assistant",
    description="Ask questions and get answers grounded only in the Bible text."
)

if __name__ == "__main__": # Bind to all interfaces so the app is accessible from Docker and deployment platforms
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
