# F1 2025 Setup Assistant Chatbot

An intelligent chatbot for F1 2025 players that provides optimized car setups, track guides, component explanations, and automatic setup adjustments based on handling feedback. Built with machine learning intent classification and retrieval-augmented generation for accurate technical responses.

## Features

- Complete car setups for all 24 F1 circuits
- ML-powered intent classification using fine-tuned DistilBERT
- Retrieval-Augmented Generation (RAG) using FAISS and sentence-transformers
- Generative responses powered by TinyLlama-1.1B
- Automatic setup adjustments based on user feedback (understeer, oversteer, tire wear, brake lock)
- Track information and racing tips
- Component explanations for F1 car setup parameters
- Web interface using Gradio

## Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support recommended (RTX 3050 or better)
- Windows OS (project developed and tested on Windows)

## Installation

1. Clone or download this repository

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU acceleration, ensure PyTorch with CUDA is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
F125/
├── data/
│   ├── raw/
│   │   ├── F125-Setups.xlsx          # Car setup data for 24 tracks
│   │   ├── processed/
│   │   │   ├── feedback_rules.json   # Rules for setup adjustments
│   │   │   ├── technical_kb.json     # Technical knowledge base
│   │   │   └── track_guides.json     # Track information
│   ├── f1-knowledge-base.txt          # F1 facts for RAG retrieval
├── models/
│   └── intent_classifier/             # Trained DistilBERT model
├── src/
│   ├── chatbot_with_generative.py    # Main chatbot class
│   ├── chatbot_gradio.py             # Gradio interface wrapper
│   ├── config.py                     # Configuration settings
│   ├── generative_layer_rag.py       # RAG-powered generation
│   ├── intent_classifier.py          # Intent classification module
│   ├── train_intent_classifier.py    # Model training script
│   └── test_intent_classifier.py     # Testing utilities
├── app.py                             # Main application entry point
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Usage

### Running the Web Interface

Start the Gradio web interface:
```bash
python app.py
```

The interface will open at http://127.0.0.1:7860

### Example Queries

**Setup Requests:**
- "Give me the setup for Monaco"
- "What setup should I use for Spa?"
- "Show me Australia setup"

**Handling Feedback:**
- "Car is understeering in slow corners"
- "I have oversteer on exit"
- "Tires are overheating"
- "Front brakes are locking"

**Track Information:**
- "Tell me about Silverstone"
- "What are the key corners at Monza?"
- "List all tracks"

**Component Information:**
- "What does front wing do?"
- "Explain differential settings"
- "How does tire pressure affect handling?"

**General Questions:**
- "How do I reduce understeer?"
- "What is DRS?"
- "Best racing line for Monaco?"

## Technical Implementation

### Intent Classification
- Fine-tuned DistilBERT model for multi-class intent classification
- 13 intent categories including setup_request, understeer_feedback, component_info, track_info, etc.
- Achieves high accuracy on F1-specific queries
- Model stored in models/intent_classifier/

### Retrieval-Augmented Generation (RAG)
- Knowledge base: f1-knowledge-base.txt with F1 facts and technical information
- Embeddings generated using sentence-transformers (all-MiniLM-L6-v2)
- FAISS index for fast similarity search
- Top-k retrieval (k=3) provides context for generation

### Language Generation
- TinyLlama-1.1B-Chat model for natural language generation
- Context-aware responses using retrieved knowledge
- Optimized for efficiency on consumer hardware
- Supports both CPU and GPU inference

### Setup Adjustment Logic
- Rule-based system for common handling issues
- Predefined adjustment mappings for understeer, oversteer, tire wear, brake lock
- Delta adjustments applied to current setup values
- Tracks last setup for iterative improvements

## Model Training

To retrain the intent classifier:

1. Prepare training data in JSON format with examples and labels
2. Run the training script:
```bash
cd src
python train_intent_classifier.py
```

The trained model will be saved to models/intent_classifier/

## Configuration

Core settings are in src/config.py:
- Data file paths
- Model paths
- Device selection (CPU/CUDA)

## Data Files

### F125-Setups.xlsx
Excel file containing setup data for all 24 circuits with columns:
- Track name
- Aerodynamics (Front Wing, Rear Wing)
- Transmission (Differential settings)
- Suspension (Springs, Anti-roll bars, Ride height)
- Brakes (Pressure, Bias)
- Tires (Pressure, Camber, Toe)

### f1-knowledge-base.txt
Plain text file with F1 technical information, formatted as paragraphs separated by blank lines. Used for RAG retrieval.

### processed/*.json
JSON files containing:
- feedback_rules.json: Rules for setup adjustments
- technical_kb.json: Component explanations
- track_guides.json: Track information and tips

## Dependencies

Key packages:
- transformers==4.38.2 - HuggingFace models
- torch>=2.0.0 - PyTorch with CUDA 11.8
- sentence-transformers - Embedding generation
- faiss-cpu - Vector similarity search
- gradio - Web interface
- pandas - Data processing
- openpyxl - Excel file handling
- scikit-learn - ML utilities

See requirements.txt for complete list.

## Performance Considerations

- GPU highly recommended for TinyLlama inference
- First load will download models (approximately 2GB)
- Intent classification is fast (<100ms per query)
- Generation time depends on max_new_tokens setting and hardware

## Known Limitations

- Knowledge base limited to manually curated F1 information
- Setup adjustments are rule-based approximations
- Generative responses quality depends on knowledge base coverage
- Windows-specific installation (bitsandbytes not supported on Windows)

## Development Notes

- Project uses PyTorch 2.7.1 with CUDA 11.8
- All cache files (__pycache__, .pyc) cleaned regularly
- Virtual environment located in src/venv/ and project root venv/
- Model quantization (4-bit) disabled due to Windows compatibility

## Future Improvements

- Expand knowledge base with more F1 technical information
- Add multi-turn conversation memory
- Implement setup export/import functionality
- Add telemetry analysis integration
- Improve generative response quality with larger models
- Add voice input/output capabilities

## License

Educational use only.

## Acknowledgments

- F1 2025 setup data from community contributors
- HuggingFace for transformer models
- Sentence-transformers for embedding models
- FAISS for efficient similarity search
