# 🏎️ F1 25 Setup Chatbot

An intelligent chatbot that helps F1 25 players optimize car setups, learn tracks, and fix handling issues across all 24 circuits.

## 📋 Features

✅ **Complete Setups** - Get optimized setups for all 24 F1 25 tracks  
✅ **Track Guides** - Learn racing lines, key corners, and track characteristics  
✅ **Component Explanations** - Understand what each setup parameter does  
✅ **Automatic Adjustments** - Report handling issues and get instant setup fixes  
✅ **Context-Aware** - Chatbot remembers your setup and adjusts iteratively  

## 🚀 Quick Start (Windows + VS Code)

### 1. Prerequisites

- Python 3.9 or higher
- VS Code installed
- 16GB RAM recommended
- RTX 3050 (4GB) or better for GPU acceleration

### 2. Setup Instructions

Open VS Code, open the `f1-chatbot` folder, then open the integrated terminal (Ctrl + `):

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Data Files

Make sure these files are in the correct locations:
- `data/raw/F125-Setups.xlsx` - Your setup data
- `data/processed/technical_kb.json` - Created automatically
- `data/processed/track_guides.json` - Created automatically
- `data/processed/feedback_rules.json` - Created automatically

### 4. Run the Chatbot

**Option A: Web Interface (Recommended)**
```bash
python app.py
```
Then open http://127.0.0.1:7860 in your browser.

**Option B: Command Line**
```bash
cd src
python chatbot.py
```

## 💬 Example Conversations

### Getting a Setup
```
You: Give me a setup for Monaco
Bot: [Shows complete setup with all parameters]

You: The car is understeering too much
Bot: [Adjusts front wing, rear wing, and differential with explanations]

You: Still a bit of understeer
Bot: [Further adjustments with suggestions]
```

### Learning About Components
```
You: What is front wing?
Bot: [Detailed explanation of front wing effects]

You: How to drive Silverstone?
Bot: [Track guide with key corners and tips]
```

### Track Tour
```
You: Tell me about all tracks
Bot: [Overview of all 24 circuits with difficulty and setup philosophy]
```

## 📁 Project Structure

```
f1-chatbot/
├── data/
│   ├── raw/
│   │   └── F125-Setups.xlsx          # Your setup data
│   └── processed/
│       ├── technical_kb.json          # Component explanations
│       ├── track_guides.json          # Track guides
│       └── feedback_rules.json        # Adjustment rules
├── models/
│   └── intent_classifier/             # Will be created during training
├── src/
│   ├── config.py                      # Configuration
│   ├── data_processor.py              # Data loading and parsing
│   ├── context_manager.py             # Conversation state tracking
│   ├── feedback_handler.py            # Setup adjustment logic
│   └── chatbot.py                     # Main orchestrator
├── app.py                             # Gradio web interface
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🎯 How It Works

### 1. Intent Detection
Currently uses **simple keyword matching**. Will be upgraded to **fine-tuned DistilBERT** for better accuracy.

### 2. Context Management
Tracks:
- Current track and setup
- Original setup for comparison
- Feedback history
- Number of adjustments made

### 3. Feedback Processing
**Rule-based system** for reliability:
- Detects issue type (understeer, oversteer, etc.)
- Applies proven racing setup adjustments
- Explains changes in user-friendly language
- Maintains setup parameter ranges

### 4. Knowledge Retrieval
Uses structured JSON knowledge bases for:
- Technical component explanations
- Track-specific racing guides
- Adjustment rules and priorities

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Enable/disable LLM-based responses (future feature)
USE_LLM_GENERATION = False

# Hardware settings
USE_4BIT_QUANTIZATION = True  # For 4GB VRAM
DEVICE = "cuda"  # Auto-detects
```

## 🧪 Testing

### Test Individual Components

```bash
# Test data processor
cd src
python data_processor.py

# Test context manager
python context_manager.py

# Test feedback handler
python feedback_handler.py
```

### Full System Test

```bash
python src/chatbot.py
```

Then try these test cases:
1. "Give me setup for Spa"
2. "Car is understeering"
3. "What is differential"
4. "How to drive Monaco"

## 📊 Current Status

### ✅ Completed (Base Model)
- ✅ Data processing pipeline
- ✅ Knowledge base system
- ✅ Context management
- ✅ Rule-based feedback handler
- ✅ Simple intent detection (keywords)
- ✅ Track name matching
- ✅ Web interface (Gradio)

### 🚧 Next Steps (Improvements)
- [ ] Fine-tune DistilBERT for intent classification
- [ ] Add Sentence-BERT for better retrieval
- [ ] Integrate Phi-3-mini for natural responses
- [ ] Create training data (intent examples)
- [ ] Add entity extraction (NER)
- [ ] Improve multi-turn conversation handling
- [ ] Add setup visualization
- [ ] Export setups to CSV/JSON

## 🎓 For NLP Project Submission

### What to Include

1. **Report** (ACL format)
   - Problem: Setup optimization assistance for racing games
   - Methodology: Multi-intent pipeline with rule-based adjustments
   - Evaluation: Intent accuracy, adjustment quality, user satisfaction
   - Novel contribution: Domain-specific racing knowledge integration

2. **Demo System**
   - Run `app.py` to show web interface
   - Prepare test conversation sequences
   - Show feedback adjustment capabilities

3. **Code & Data**
   - GitHub repository (organized and documented)
   - Training data for intent classifier (when created)
   - Pre-trained models
   - Setup data and knowledge bases

4. **Evaluation**
   - Intent classification accuracy
   - User feedback on adjustment quality
   - Conversation coherence metrics
   - Comparison: rule-based vs LLM-based

## 🤝 Development Workflow

### Phase 1: Base System (Current)
Week 1: ✅ Data prep, knowledge bases, core logic

### Phase 2: ML Integration (Next)
Week 2:
- Create intent training data (100+ examples per intent)
- Fine-tune DistilBERT for classification
- Test and evaluate accuracy

Week 3:
- Add Sentence-BERT for retrieval
- Integrate Phi-3-mini (optional for natural responses)
- Fine-tune with LoRA

### Phase 3: Polish & Testing
Week 4:
- User testing and feedback
- Bug fixes and improvements
- Documentation and demo prep

## 💡 Tips

### For Best Results
1. **Be specific** with feedback ("understeer in slow corners" vs "bad handling")
2. **Iterate gradually** - make small adjustments and test
3. **Reset context** when switching tracks
4. **Save good setups** - copy the output when you find a good balance

### Common Issues
- **"Setup not found"** - Check track name spelling (use autocomplete)
- **"No adjustments"** - Make sure you've loaded a setup first
- **Memory errors** - Close other applications, reduce batch size

## 📝 License

Educational project for NLP coursework.

## 🙏 Acknowledgments

- Setup data from F1 25 community
- Track information from official F1 sources
- Racing setup knowledge from Driver61, Chris Haye, and community guides

---

**Ready to optimize your setups? Run `python app.py` and start chatting!** 🏁
