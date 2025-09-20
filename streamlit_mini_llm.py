"""
Streamlit UI for Mini Transformer Language Model
A web interface for training and interacting with a mini transformer model
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import re
import io
import time
import pickle
from typing import List, Tuple
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Mini Transformer LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE MODEL CODE (Same as before, but streamlined for Streamlit)
# ============================================================================

class Config:
    """Configuration class with default hyperparameters"""
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    num_layers = 3
    max_seq_len = 64
    dropout = 0.1
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 50
    device = 'cpu'
    max_gen_length = 100
    temperature = 0.8

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        
        self.vocab[self.pad_token] = 0
        self.vocab[self.unk_token] = 1
        self.vocab[self.eos_token] = 2
        self.next_id = 3
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 1000):
        word_counts = {}
        
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:max_vocab_size - 3]:
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
        
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return words
    
    def encode(self, text: str) -> List[int]:
        words = self._tokenize_text(text)
        token_ids = []
        for word in words:
            token_ids.append(self.vocab.get(word, self.vocab[self.unk_token]))
        token_ids.append(self.vocab[self.eos_token])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        words = []
        for token_id in token_ids:
            if token_id == self.vocab[self.eos_token]:
                break
            word = self.inverse_vocab.get(token_id, self.unk_token)
            if word not in [self.pad_token, self.unk_token]:
                words.append(word)
        return ' '.join(words)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sequences = []
        
        for text in texts:
            token_ids = tokenizer.encode(text)
            
            # If the text is shorter than seq_len, pad it or use what we have
            if len(token_ids) <= seq_len:
                # Pad short sequences
                padded_seq = token_ids + [tokenizer.vocab[tokenizer.pad_token]] * (seq_len + 1 - len(token_ids))
                if len(padded_seq) >= seq_len + 1:
                    self.sequences.append(padded_seq[:seq_len + 1])
            else:
                # Create overlapping sequences for longer texts
                step_size = max(1, seq_len // 4)  # Smaller step for more data
                for i in range(0, len(token_ids) - seq_len, step_size):
                    seq = token_ids[i:i + seq_len + 1]
                    if len(seq) == seq_len + 1:
                        self.sequences.append(seq)
        
        # If still no sequences, create at least one from combined text
        if len(self.sequences) == 0:
            combined_text = " ".join(texts)
            combined_ids = tokenizer.encode(combined_text)
            
            if len(combined_ids) < seq_len:
                # Pad to minimum required length
                padded_ids = combined_ids + [tokenizer.vocab[tokenizer.pad_token]] * (seq_len + 1 - len(combined_ids))
                self.sequences.append(padded_ids[:seq_len + 1])
            else:
                # Use first seq_len+1 tokens
                self.sequences.append(combined_ids[:seq_len + 1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(pos_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits

def generate_text(model: MiniTransformer, tokenizer: SimpleTokenizer, 
                 prompt: str, max_length: int = 50, temperature: float = 0.8, 
                 device: str = 'cpu') -> str:
    model.eval()
    model = model.to(device)
    
    # Encode prompt and handle empty prompts
    if not prompt.strip():
        prompt = "the"
    
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) == 0:
        input_ids = [tokenizer.vocab.get('the', tokenizer.vocab[tokenizer.unk_token])]
    
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Use only the last max_seq_len tokens to stay within model limits
            context_len = min(len(generated_ids), model.max_seq_len)
            current_input = torch.tensor([generated_ids[-context_len:]], dtype=torch.long).to(device)
            
            # Make sure input doesn't exceed model's positional embedding size
            if current_input.size(1) > model.max_seq_len:
                current_input = current_input[:, -model.max_seq_len:]
            
            try:
                logits = model(current_input)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Add small epsilon to prevent issues with zero probabilities
                next_token_logits = next_token_logits + 1e-8
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if we generate end-of-sequence token
                if next_token == tokenizer.vocab[tokenizer.eos_token]:
                    break
                
                # Stop if we generate padding token (shouldn't generate this)
                if next_token == tokenizer.vocab[tokenizer.pad_token]:
                    continue
                    
                generated_ids.append(next_token)
                
            except Exception as e:
                print(f"Generation error: {e}")
                break
    
    return tokenizer.decode(generated_ids)

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🤖 Mini Transformer Language Model")
    st.markdown("Train and interact with a tiny transformer model!")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Hyperparameters
    embed_dim = st.sidebar.slider("Embedding Dimension", 64, 256, 128, 32)
    num_heads = st.sidebar.selectbox("Number of Attention Heads", [2, 4, 8], index=1)
    num_layers = st.sidebar.slider("Number of Transformer Layers", 1, 6, 3)
    max_seq_len = st.sidebar.slider("Max Sequence Length", 16, 128, 32, 16)  # Reduced default
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)
    
    st.sidebar.header("🎯 Training Configuration")
    batch_size = st.sidebar.slider("Batch Size", 4, 32, 16, 4)
    learning_rate = st.sidebar.select_slider("Learning Rate", 
                                           options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], 
                                           value=1e-3, format_func=lambda x: f"{x:.0e}")
    num_epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Data & Training", "🎮 Text Generation", "📊 Model Info", "💾 Save/Load"])
    
    with tab1:
        st.header("Training Data")
        
        # Data input options
        data_source = st.radio("Choose data source:", 
                              ["Sample Texts", "Upload File", "Custom Text"])
        
        training_texts = []
        
        if data_source == "Sample Texts":
            st.info("Using built-in sample texts for demonstration")
            training_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "To be or not to be, that is the question.",
                "In the beginning was the Word, and the Word was with God.",
                "It was the best of times, it was the worst of times.",
                "All happy families are alike; each unhappy family is unhappy in its own way.",
                "Call me Ishmael. Some years ago—never mind how long precisely.",
                "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
                "Space: the final frontier. These are the voyages of the starship Enterprise.",
                "A long time ago in a galaxy far, far away, there was a great adventure.",
                "The only way to do great work is to love what you do.",
                "I have a dream that one day this nation will rise up.",
                "Ask not what your country can do for you, ask what you can do for your country.",
                "That's one small step for man, one giant leap for mankind.",
                "The future belongs to those who believe in the beauty of their dreams."
            ]
        
        elif data_source == "Upload File":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                content = str(uploaded_file.read(), "utf-8")
                # Split by sentences or paragraphs
                training_texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.success(f"Loaded {len(training_texts)} lines from file")
                st.text_area("Preview:", value=content[:500] + "..." if len(content) > 500 else content, height=100)
        
        elif data_source == "Custom Text":
            custom_text = st.text_area("Enter your training text (separate sentences/paragraphs with new lines):", 
                                     height=200,
                                     placeholder="Enter your text here...\nEach line will be treated as a separate training example.")
            if custom_text:
                training_texts = [line.strip() for line in custom_text.split('\n') if line.strip()]
                st.info(f"Ready to train on {len(training_texts)} text samples")
        
        # Training section
        if training_texts:
            st.subheader("Model Training")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    # Validate training data
                    if not training_texts:
                        st.error("No training data available! Please provide some text.")
                        return
                    
                    # Check if texts are too short
                    total_chars = sum(len(text) for text in training_texts)
                    if total_chars < 100:
                        st.warning("Training data is very short. Consider adding more text for better results.")
                    
                    # Create progress bars
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_placeholder = st.empty()
                    
                    try:
                        # Initialize model and tokenizer
                        tokenizer = SimpleTokenizer()
                        tokenizer.build_vocab(training_texts, max_vocab_size=1000)
                        
                        # Create dataset and check if it has sequences
                        dataset = TextDataset(training_texts, tokenizer, max_seq_len)
                        
                        if len(dataset) == 0:
                            st.error("❌ Could not create training sequences. Try:")
                            st.write("- Adding longer texts")
                            st.write("- Reducing the 'Max Sequence Length' in the sidebar")
                            st.write("- Adding more training examples")
                            return
                        
                        st.info(f"Created {len(dataset)} training sequences from {len(training_texts)} texts")
                        
                        # Adjust batch size if needed
                        actual_batch_size = min(batch_size, len(dataset))
                        if actual_batch_size != batch_size:
                            st.info(f"Adjusted batch size from {batch_size} to {actual_batch_size} (limited by dataset size)")
                        
                        dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)
                        
                        model = MiniTransformer(
                            vocab_size=len(tokenizer.vocab),
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            max_seq_len=max_seq_len,
                            dropout=dropout
                        )
                        
                        # Training setup
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = nn.CrossEntropyLoss()
                        
                        model.train()
                        training_history = []
                        
                        # Training loop
                        for epoch in range(num_epochs):
                            total_loss = 0
                            num_batches = len(dataloader)
                            
                            for batch_idx, (inputs, targets) in enumerate(dataloader):
                                logits = model(inputs)
                                logits = logits.view(-1, logits.size(-1))
                                targets = targets.view(-1)
                                
                                loss = criterion(logits, targets)
                                
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                
                                total_loss += loss.item()
                            
                            avg_loss = total_loss / num_batches
                            training_history.append(avg_loss)
                            
                            # Update progress
                            progress = (epoch + 1) / num_epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
                            
                            # Show loss plot
                            if len(training_history) > 1:
                                df = pd.DataFrame({'Epoch': range(1, len(training_history)+1), 
                                                 'Loss': training_history})
                                fig = px.line(df, x='Epoch', y='Loss', title='Training Loss')
                                loss_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Save to session state
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.training_history = training_history
                        st.session_state.model_trained = True
                        
                        st.success("✅ Training completed!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.write("**Possible solutions:**")
                        st.write("- Try reducing the sequence length")
                        st.write("- Add more/longer training texts")
                        st.write("- Reduce batch size")
                        st.exception(e)
            
            with col2:
                if st.session_state.model_trained:
                    st.success("Model is trained and ready!")
                    total_params = sum(p.numel() for p in st.session_state.model.parameters())
                    st.metric("Model Parameters", f"{total_params:,}")
                    st.metric("Vocabulary Size", len(st.session_state.tokenizer.vocab))
                else:
                    st.info("No trained model available")
    
    with tab2:
        st.header("Text Generation")
        
        if not st.session_state.model_trained:
            st.warning("Please train a model first in the 'Data & Training' tab!")
        else:
            # Generation parameters
            col1, col2 = st.columns([2, 1])
            
            with col1:
                prompt = st.text_input("Enter your prompt:", 
                                     value="The quick brown",
                                     help="Start typing to generate text based on your trained model")
            
            with col2:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1,
                                      help="Higher = more creative, Lower = more focused")
                max_length = st.slider("Max Length", 10, 200, 100, 10)
            
            if st.button("🎯 Generate Text", type="primary"):
                if prompt.strip():
                    with st.spinner("Generating text..."):
                        generated_text = generate_text(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            prompt,
                            max_length=max_length,
                            temperature=temperature
                        )
                    
                    st.subheader("Generated Text:")
                    st.write(f"**Prompt:** {prompt}")
                    st.write(f"**Generated:** {generated_text}")
                    
                    # Copy button
                    st.code(generated_text, language=None)
                else:
                    st.error("Please enter a prompt!")
            
            # Sample prompts
            st.subheader("Try these sample prompts:")
            sample_prompts = [
                "The quick brown",
                "To be or not",
                "In the beginning",
                "It was the best"
            ]
            
            cols = st.columns(len(sample_prompts))
            for i, sample_prompt in enumerate(sample_prompts):
                with cols[i]:
                    if st.button(f'"{sample_prompt}"', key=f"sample_{i}"):
                        # Generate text for sample prompt
                        with st.spinner("Generating..."):
                            generated_text = generate_text(
                                st.session_state.model,
                                st.session_state.tokenizer,
                                sample_prompt,
                                max_length=max_length,
                                temperature=temperature
                            )
                        st.write(f"**{sample_prompt}** → {generated_text}")
    
    with tab3:
        st.header("Model Information")
        
        if st.session_state.model_trained:
            model = st.session_state.model
            tokenizer = st.session_state.tokenizer
            
            # Model architecture
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Architecture")
                st.metric("Embedding Dimension", model.embed_dim)
                st.metric("Number of Layers", len(model.blocks))
                st.metric("Number of Attention Heads", model.blocks[0].attention.num_heads)
                st.metric("Max Sequence Length", model.max_seq_len)
                
                total_params = sum(p.numel() for p in model.parameters())
                st.metric("Total Parameters", f"{total_params:,}")
            
            with col2:
                st.subheader("Vocabulary Info")
                st.metric("Vocabulary Size", len(tokenizer.vocab))
                st.metric("Special Tokens", 3)
                
                # Show some vocabulary words
                vocab_sample = list(tokenizer.vocab.keys())[:20]
                st.write("**Sample Vocabulary:**")
                st.write(", ".join(vocab_sample))
            
            # Training history
            if st.session_state.training_history:
                st.subheader("Training History")
                df = pd.DataFrame({
                    'Epoch': range(1, len(st.session_state.training_history)+1),
                    'Loss': st.session_state.training_history
                })
                
                fig = px.line(df, x='Epoch', y='Loss', 
                            title='Training Loss Over Time',
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Final Training Loss", f"{st.session_state.training_history[-1]:.4f}")
        else:
            st.info("Train a model to see detailed information here.")
    
    with tab4:
        st.header("Save/Load Model")
        
        if st.session_state.model_trained:
            st.subheader("💾 Save Model")
            
            if st.button("Download Model", type="primary"):
                # Create a dictionary with model and tokenizer
                model_data = {
                    'model_state_dict': st.session_state.model.state_dict(),
                    'tokenizer': st.session_state.tokenizer,
                    'config': {
                        'vocab_size': len(st.session_state.tokenizer.vocab),
                        'embed_dim': st.session_state.model.embed_dim,
                        'num_heads': st.session_state.model.blocks[0].attention.num_heads,
                        'num_layers': len(st.session_state.model.blocks),
                        'max_seq_len': st.session_state.model.max_seq_len,
                    },
                    'training_history': st.session_state.training_history
                }
                
                # Serialize to bytes
                buffer = io.BytesIO()
                torch.save(model_data, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="📁 Download Model File",
                    data=buffer.getvalue(),
                    file_name="mini_transformer_model.pt",
                    mime="application/octet-stream"
                )
        
        st.subheader("📂 Load Model")
        uploaded_model = st.file_uploader("Upload a saved model file", type=['pt'])
        
        if uploaded_model is not None:
            if st.button("Load Model"):
                try:
                    # Load the model data
                    model_data = torch.load(uploaded_model, map_location='cpu')
                    
                    # Reconstruct the model
                    config = model_data['config']
                    model = MiniTransformer(
                        vocab_size=config['vocab_size'],
                        embed_dim=config['embed_dim'],
                        num_heads=config['num_heads'],
                        num_layers=config['num_layers'],
                        max_seq_len=config['max_seq_len']
                    )
                    model.load_state_dict(model_data['model_state_dict'])
                    
                    # Update session state
                    st.session_state.model = model
                    st.session_state.tokenizer = model_data['tokenizer']
                    st.session_state.training_history = model_data.get('training_history', [])
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    main()