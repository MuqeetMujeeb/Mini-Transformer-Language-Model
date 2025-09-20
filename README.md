# 🤖 Mini Transformer Language Model

A **simple, educational implementation** of a transformer-based language model that you can train and run on your laptop! Perfect for learning how modern AI language models work.

## ✨ What is this?

This project contains a **minimal version of ChatGPT/GPT** - a transformer language model that can:
- Learn patterns from text data
- Generate new text based on prompts
- Run entirely on CPU (no GPU needed!)
- Be trained in minutes, not hours

**Two versions included:**
- 📝 **Command-line script** (`mini_transformer.py`) - Pure Python implementation
- 🌐 **Streamlit web app** (`streamlit_app.py`) - Beautiful web interface

## 🎯 Why use this?

- **Educational**: Understand how transformers work with clear, commented code
- **Lightweight**: Trains in minutes on your laptop
- **Customizable**: Easy to modify architecture and hyperparameters
- **Complete**: From tokenization to text generation in one script
- **Visual**: Web interface with training charts and interactive generation

## 🚀 Quick Start

### Option 1: Command Line Version

```bash
# Install PyTorch
pip install torch

# Run the script
python mini_transformer.py
```

The script will:
1. Train on sample texts
2. Show training progress  
3. Generate sample text
4. Start interactive mode for custom prompts

### Option 2: Web Interface (Recommended)

```bash
# Install dependencies
pip install streamlit torch plotly pandas

# Launch web app
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 🎮 How to Use

### Web Interface Features:

#### 📚 **Data & Training Tab**
- Choose your data source:
  - **Sample texts** (built-in examples)
  - **Upload file** (your own .txt file)  
  - **Custom text** (paste directly)
- Adjust model settings (size, layers, attention heads)
- Click "Start Training" and watch real-time progress
- See loss curves update live

#### 🎯 **Text Generation Tab**  
- Enter any prompt
- Adjust creativity (temperature) and length
- Generate text instantly
- Try sample prompts with one click

#### 📊 **Model Info Tab**
- View model architecture details
- Inspect vocabulary  
- See training history charts

#### 💾 **Save/Load Tab**
- Download trained models
- Upload previously saved models
- Share models with others

## 🏗️ Model Architecture

Our mini transformer includes all the key components:

```
Input Text → Tokenizer → Embeddings → Transformer Blocks → Output
```

**Transformer Block:**
- Multi-head self-attention (learns relationships between words)
- Feed-forward network (processes information)  
- Residual connections & layer normalization (training stability)
- Causal masking (prevents looking at future words)

**Default Settings:**
- 128 embedding dimensions
- 4 attention heads  
- 3 transformer layers
- ~50K parameters (vs GPT-3's 175B!)


## 🔧 Customization

### Easy Changes in Web Interface:
- **Model size**: Adjust embedding dimensions (64-512)
- **Complexity**: Change number of layers (1-8) and attention heads (2-16)
- **Training**: Modify learning rate, batch size, epochs
- **Generation**: Control creativity with temperature (0.1-2.0)

### Code Modifications:
- **Tokenizer**: Switch from word-level to character-level in `SimpleTokenizer`
- **Architecture**: Add more transformer blocks or change dimensions
- **Training**: Implement different optimizers or learning rate schedules
- **Generation**: Add beam search or top-k sampling

## 📊 Performance Examples

**Training on sample data (14 sentences):**
- Training time: ~2 minutes on laptop CPU  
- Final loss: ~2.5
- Vocabulary: ~100 unique words
- Model size: ~50K parameters

**Sample generations:**
- Prompt: `"The quick brown"` → `"The quick brown fox jumps over the lazy dog and runs through the forest"`
- Prompt: `"To be or"` → `"To be or not to be that is the question of life"`

## 🛠️ Requirements

- **Python 3.7+**
- **PyTorch** (CPU version is fine)
- **Streamlit** (for web interface)
- **Plotly & Pandas** (for charts)

```bash
pip install torch streamlit plotly pandas
```

## 📚 Learning Resources

This implementation is designed for education. Here's what you'll learn:

1. **Tokenization**: How text becomes numbers
2. **Embeddings**: How words become vectors  
3. **Attention**: How models focus on relevant words
4. **Transformers**: The architecture powering ChatGPT
5. **Training**: How models learn from data
6. **Generation**: How models create new text

## 🎓 Educational Use Cases

- **Students**: Learn transformer architecture hands-on
- **Teachers**: Demonstrate how AI language models work
- **Developers**: Prototype ideas before scaling up
- **Researchers**: Quick experiments with new architectures
- **Hobbyists**: Build your own mini-ChatGPT!

## 🤝 Contributing

Contributions welcome! Ideas for improvements:

- [ ] Add beam search for better generation
- [ ] Implement character-level tokenization
- [ ] Add model quantization for smaller size
- [ ] Create more example datasets
- [ ] Add evaluation metrics
- [ ] Implement fine-tuning capabilities

## 📝 License

MIT License - feel free to use, modify, and share!

## 🙏 Acknowledgments

Inspired by:
- "Attention Is All You Need" (Vaswani et al.)
- Andrej Karpathy's educational materials
- The transformer architecture from OpenAI's GPT models

---

### 🚀 Ready to build your own language model? 

Clone this repo and start training in minutes!

```bash
git clone https://github.com/MuqeetMujeeb/mini-transformer-llm
cd mini-transformer-llm
pip install -r requirements.txt
streamlit run streamlit_app.py
```
