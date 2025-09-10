import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from typing import Dict, List, Tuple

class BERTForDualTruthfulness(nn.Module):
    """BERT model with dual classification heads for token and sentence level truthfulness"""
    
    def __init__(self, bert_model, hidden_size=768):
        super(BERTForDualTruthfulness, self).__init__()
        self.bert = bert_model
        self.dual_classifier = DualTruthfulnessClassifier(hidden_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        token_logits, sentence_logits = self.dual_classifier(sequence_output)
        
        return token_logits, sentence_logits, outputs

class DualTruthfulnessClassifier(nn.Module):
    """Dual classifier for both token-level and sentence-level truthfulness detection"""
    
    def __init__(self, hidden_size=768, num_classes=2):
        super(DualTruthfulnessClassifier, self).__init__()
        
        # Token-level classifier
        self.token_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Sentence-level classifier (uses CLS token)
        self.sentence_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, sequence_output):
        # Token-level predictions for all tokens
        token_logits = self.token_classifier(sequence_output)
        
        # Sentence-level prediction using CLS token (first token)
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        sentence_logits = self.sentence_classifier(cls_output)
        
        return token_logits, sentence_logits

class SteganographicDetector:
    """Base class for steganographic detection in text"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def detect(self, text: str) -> Dict[str, float]:
        """
        Detect steganographic content in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            token_logits, sentence_logits, outputs = self.model(**inputs)
            
            # Get probabilities
            token_probs = torch.softmax(token_logits, dim=-1)
            sentence_probs = torch.softmax(sentence_logits, dim=-1)
            
            # Extract steganographic probabilities (class 1)
            token_steg_probs = token_probs[:, :, 1].cpu().numpy()
            sentence_steg_prob = sentence_probs[:, 1].cpu().numpy()[0]
            
            # Calculate statistics
            mean_token_prob = np.mean(token_steg_probs)
            max_token_prob = np.max(token_steg_probs)
            token_variance = np.var(token_steg_probs)
            
            # Simple risk assessment
            risk_score = 0.6 * sentence_steg_prob + 0.4 * mean_token_prob
            
            # Determine classification
            classification = "STEGANOGRAPHIC" if risk_score > 0.5 else "NORMAL"
            
            results = {
                'risk_score': float(risk_score),
                'classification': classification,
                'sentence_probability': float(sentence_steg_prob),
                'mean_token_probability': float(mean_token_prob),
                'max_token_probability': float(max_token_prob),
                'token_variance': float(token_variance),
                'token_probabilities': token_steg_probs.tolist(),
                'analysis': self._generate_analysis(risk_score, token_variance, sentence_steg_prob)
            }
            
        return results
    
    def _generate_analysis(self, risk_score: float, token_variance: float, sentence_prob: float) -> List[str]:
        """Generate human-readable analysis of detection results"""
        analysis = []
        
        if risk_score > 0.7:
            analysis.append("High probability of steganographic content detected")
        elif risk_score > 0.3:
            analysis.append("Moderate steganographic indicators present")
        else:
            analysis.append("Low steganographic probability")
            
        if token_variance < 0.01:
            analysis.append(f"Unnaturally consistent token variance ({token_variance:.4f})")
        elif token_variance > 0.2:
            analysis.append(f"High token variance suggests complex encoding ({token_variance:.4f})")
            
        if sentence_prob > 0.8:
            analysis.append("Strong sentence-level steganographic indicators")
        elif sentence_prob > 0.6:
            analysis.append("Moderate sentence-level anomalies detected")
            
        return analysis
    
    def batch_detect(self, texts: List[str]) -> List[Dict[str, float]]:
        """Detect steganographic content in multiple texts"""
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)
        return results
    
    def extract_embeddings(self, text: str, layer_idx: int = -1) -> np.ndarray:
        """Extract embeddings from a specific layer"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get embeddings from specified layer
            if layer_idx == -1:
                layer_idx = len(hidden_states) - 1
                
            embeddings = hidden_states[layer_idx][:, 0, :].cpu().numpy()  # CLS token
            
        return embeddings.flatten()
    
    def analyze_attention_patterns(self, text: str) -> Dict[str, np.ndarray]:
        """Analyze attention patterns for steganographic indicators"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs, output_attentions=True)
            attentions = outputs.attentions
            
            # Process attention weights
            attention_analysis = {}
            
            for layer_idx, attention in enumerate(attentions):
                # attention shape: [batch_size, num_heads, seq_len, seq_len]
                attention_weights = attention[0].cpu().numpy()  # First batch item
                
                # Calculate attention statistics
                mean_attention = np.mean(attention_weights, axis=0)  # Average across heads
                attention_entropy = -np.sum(mean_attention * np.log(mean_attention + 1e-10), axis=-1)
                
                attention_analysis[f'layer_{layer_idx}'] = {
                    'mean_attention': mean_attention,
                    'attention_entropy': attention_entropy,
                    'max_attention': np.max(attention_weights),
                    'attention_variance': np.var(attention_weights)
                }
                
        return attention_analysis
