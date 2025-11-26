"""
Gradio Web Interface for F1 Chatbot
Run this to get a web UI for testing
"""
import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from chatbot_with_generative import F125ChatbotWithGenerative


# Initialize chatbot (global for session persistence)
bot = F125ChatbotWithGenerative()


def chat_interface(message, history):
    """
    Gradio chat interface function
    
    Args:
        message: Current user message
        history: List of [user_msg, bot_msg] pairs
    
    Returns:
        Updated history
    """
    
    # Get response from bot
    response = bot.chat(message)
    
    # Append to history
    history.append((message, response))
    
    return history, history


def reset_conversation():
    """Reset the chatbot context"""
    return [], []


# Create Gradio interface
with gr.Blocks(title="F1 25 Setup Chatbot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🏎️ F1 25 Setup Chatbot
        
        Get setups, track guides, and setup adjustments for all 24 F1 circuits!
        
        **Try these commands:**
        - "Give me a setup for Monaco"
        - "How to drive Silverstone"
        - "What is front wing"
        - "The car is understeering"
        - "Tell me about all tracks"
        """
    )
    
    chatbot_ui = gr.Chatbot(
        value=[],
        height=500,
        label="Chat History"
    )
    
    with gr.Row():
        with gr.Column(scale=9):
            msg_box = gr.Textbox(
                placeholder="Ask me about setups, tracks, or report handling issues...",
                label="Your Message",
                lines=1
            )
        with gr.Column(scale=1):
            send_btn = gr.Button("Send", variant="primary")
    
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Conversation")
        gr.Markdown("*Tip: Reset if you want to start fresh with a new track*")
    
    # Examples
    gr.Examples(
        examples=[
            "Give me a setup for Monaco",
            "How to drive Spa",
            "What is differential",
            "The car is understeering too much",
            "Tell me about all tracks",
            "Car is oversteering on corner exit",
            "What are tire pressures"
        ],
        inputs=msg_box
    )
    
    # Event handlers
    msg_box.submit(
        chat_interface,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, chatbot_ui]
    ).then(
        lambda: "",  # Clear input box
        outputs=msg_box
    )
    
    send_btn.click(
        chat_interface,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, chatbot_ui]
    ).then(
        lambda: "",
        outputs=msg_box
    )
    
    reset_btn.click(
        reset_conversation,
        outputs=[chatbot_ui, msg_box]
    )
    
    gr.Markdown(
        """
        ---
        ### Features:
        - ✅ Complete setups for all 24 tracks
        - ✅ Track guides with racing tips
        - ✅ Setup component explanations
        - ✅ Automatic setup adjustments based on feedback
        - ✅ Context-aware conversation
        
        ### How to Use:
        1. Ask for a setup for any track
        2. Try it and report issues (understeer, oversteer, etc.)
        3. Bot will adjust your setup automatically
        4. Keep iterating until perfect!
        """
    )


if __name__ == "__main__":
    print("\n🏎️ Starting F1 25 Setup Chatbot Web Interface...")
    print("🌐 Opening in browser...\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
