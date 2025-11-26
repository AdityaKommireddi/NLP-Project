import gradio as gr
from chatbot_with_generative import F125ChatbotWithGenerative

# Load the chatbot
bot = F125ChatbotWithGenerative()

def chatbot_interface(message, history):
    # Return bot response
    return bot.chat(message)

demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="F1 2025 Setup and Racing Assistant",
    description="Just ask for setups and F1 car help!",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
