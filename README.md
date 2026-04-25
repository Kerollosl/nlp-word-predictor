# NLP Word Predictor 📝

## 📋 Overview

A natural language processing project that predicts the next word in a sequence using LSTM neural networks trained on the King James Bible. The model learns language patterns and can generate contextually appropriate word predictions.

## 🎯 Features

- **Next Word Prediction:** Predicts the most likely next word given a sequence
- **Text Generation:** Can generate coherent text sequences
- **Pre-trained Model:** Includes saved model for immediate use
- **Custom Training:** Retrain on any text corpus

## 🧠 Model Architecture

- **Type:** LSTM (Long Short-Term Memory) Neural Network
- **Training Data:** King James Bible (4.4 MB text corpus)
- **Tokenization:** Word-level tokenization
- **Sequence Length:** Configurable context window

## 📊 Dataset

- **File:** `bible.txt` (4,451,368 bytes)
- **Source:** King James Bible
- **Vocabulary Size:** Thousands of unique words
- **Training Approach:** Sequential word prediction

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow numpy pandas
```

### Training a New Model
```bash
python word_predictor.py
```

### Using Pre-trained Model
```bash
python load_model_and_predict.py
```

## 📁 Repository Contents

### Python Scripts
- `word_predictor.py` - Main training script for the LSTM model
- `load_model_and_predict.py` - Load saved model and make predictions
- `functions.py` - Utility functions for text processing and model operations

### Data Files
- `bible.txt` - Training corpus (King James Bible)

### Documentation
- `ReadMe.txt` - Original project documentation

## 🛠️ Technical Stack

- **Deep Learning:** TensorFlow/Keras
- **NLP:** Custom tokenization and sequence processing
- **Data Processing:** NumPy, Pandas
- **Model:** LSTM with embedding layer

## 📝 Usage Examples

### Predict Next Word
```python
# Load model and predict
from load_model_and_predict import predict_next_word

sequence = "In the beginning"
next_word = predict_next_word(sequence)
print(f"Next word: {next_word}")
```

### Train Custom Model
```python
# Train on your own text
from word_predictor import train_model

train_model(text_file='your_text.txt', 
           epochs=50,
           sequence_length=10)
```

## 🎓 Applications

- Text autocompletion
- Writing assistance
- Language modeling research
- Biblical text analysis
- Educational NLP demonstrations

## 🔧 Configuration

Key parameters in `word_predictor.py`:
- **Sequence Length:** Number of words to use as context
- **Epochs:** Training iterations
- **Batch Size:** Training batch size
- **Embedding Dimension:** Word vector size
- **LSTM Units:** Hidden layer size

## 📈 Model Performance

The model learns:
- Biblical language patterns
- Archaic English structure
- Common word sequences
- Contextual word relationships

