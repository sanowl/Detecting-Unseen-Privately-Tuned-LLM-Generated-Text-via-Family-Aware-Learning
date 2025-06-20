"""
Adversarial Robustness for PhantomHunter
Implements state-of-the-art adversarial attacks and defenses based on latest research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import math
import random
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForMaskedLM, T5ForConditionalGeneration, T5Tokenizer
import scipy.stats as stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

class TextAttacks:
    """Sophisticated text attack methods based on recent adversarial NLP research"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_models()
        self._load_linguistic_resources()
        
    def _load_models(self):
        """Load models for sophisticated paraphrasing attacks"""
        try:
            # T5 for paraphrasing
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
            
            # BERT for masked language modeling attacks
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(self.device)
            
        except Exception as e:
            print(f"Warning: Could not load paraphrasing models: {e}")
            self.t5_model = None
            self.bert_model = None
    
    def _load_linguistic_resources(self):
        """Load linguistic resources for attacks"""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            
            # Load word frequency data for realistic substitutions
            self.word_freq = self._load_word_frequencies()
            
        except Exception as e:
            print(f"Warning: Could not load linguistic resources: {e}")
            self.stop_words = set()
            self.word_freq = defaultdict(float)
    
    def _load_word_frequencies(self) -> Dict[str, float]:
        """Load word frequency data for realistic attacks"""
        # Simplified word frequency - in production, use real frequency data
        common_words = {
            'the': 0.069, 'of': 0.036, 'and': 0.028, 'a': 0.025, 'to': 0.025,
            'in': 0.017, 'is': 0.013, 'it': 0.013, 'you': 0.013, 'that': 0.012,
            'he': 0.011, 'was': 0.011, 'for': 0.011, 'on': 0.010, 'are': 0.009,
            'as': 0.009, 'with': 0.009, 'his': 0.008, 'they': 0.008, 'i': 0.008
        }
        return defaultdict(lambda: 0.001, common_words)
    
    def textfooler_attack(self, text: str, target_model, max_substitutions: int = 10,
                         similarity_threshold: float = 0.8) -> str:
        """
        TextFooler attack: Replace words with synonyms that maintain semantic similarity
        Based on "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification"
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Get original prediction
        original_pred = self._get_model_prediction(target_model, text)
        
        # Compute word importance scores
        word_importance = self._compute_word_importance(target_model, text, tokens)
        
        # Sort words by importance
        sorted_indices = sorted(range(len(tokens)), 
                              key=lambda i: word_importance[i], reverse=True)
        
        modified_tokens = tokens.copy()
        substitutions = 0
        
        for idx in sorted_indices:
            if substitutions >= max_substitutions:
                break
                
            token = tokens[idx]
            pos = pos_tags[idx][1]
            
            # Skip if word is not suitable for substitution
            if (token in self.stop_words or 
                not token.isalpha() or 
                len(token) < 3):
                continue
            
            # Get candidate substitutions
            candidates = self._get_synonym_candidates(token, pos)
            
            # Test each candidate
            for candidate in candidates:
                if candidate == token:
                    continue
                    
                # Create modified text
                test_tokens = modified_tokens.copy()
                test_tokens[idx] = candidate
                test_text = ' '.join(test_tokens)
                
                # Check semantic similarity
                if self._semantic_similarity(text, test_text) < similarity_threshold:
                    continue
                
                # Check if attack succeeds
                new_pred = self._get_model_prediction(target_model, test_text)
                if new_pred != original_pred:
                    modified_tokens[idx] = candidate
                    substitutions += 1
                    break
        
        return ' '.join(modified_tokens)
    
    def bert_attack(self, text: str, target_model, num_candidates: int = 10) -> str:
        """
        BERT-based attack using masked language modeling
        Based on "BERT-ATTACK: Adversarial Attack Against BERT Using BERT"
        """
        if self.bert_model is None:
            return self._fallback_synonym_attack(text)
        
        tokens = self.bert_tokenizer.tokenize(text)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        
        # Compute word importance
        word_importance = self._compute_bert_word_importance(target_model, text, tokens)
        
        # Sort positions by importance
        sorted_positions = sorted(range(len(tokens)), 
                                key=lambda i: word_importance[i], reverse=True)
        
        modified_tokens = tokens.copy()
        
        for pos in sorted_positions:
            if tokens[pos].startswith('##') or not tokens[pos].isalpha():
                continue
            
            # Get BERT predictions for masked position
            candidates = self._get_bert_candidates(tokens, pos, num_candidates)
            
            # Test each candidate
            for candidate in candidates:
                test_tokens = modified_tokens.copy()
                test_tokens[pos] = candidate
                test_text = self.bert_tokenizer.convert_tokens_to_string(test_tokens)
                
                # Check if attack succeeds
                original_pred = self._get_model_prediction(target_model, text)
                new_pred = self._get_model_prediction(target_model, test_text)
                
                if new_pred != original_pred:
                    modified_tokens[pos] = candidate
                    break
        
        return self.bert_tokenizer.convert_tokens_to_string(modified_tokens)
    
    def deepwordbug_attack(self, text: str, edit_rate: float = 0.1) -> str:
        """
        DeepWordBug attack: Character-level modifications
        Based on "Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers"
        """
        tokens = text.split()
        num_edits = max(1, int(len(tokens) * edit_rate))
        
        # Character-level edit operations
        operations = ['insert', 'delete', 'substitute', 'swap']
        
        modified_tokens = tokens.copy()
        
        for _ in range(num_edits):
            # Choose random token to modify
            token_idx = random.randint(0, len(modified_tokens) - 1)
            token = modified_tokens[token_idx]
            
            if len(token) < 2:
                continue
            
            operation = random.choice(operations)
            
            if operation == 'insert':
                # Insert random character
                pos = random.randint(0, len(token))
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                modified_token = token[:pos] + char + token[pos:]
                
            elif operation == 'delete':
                # Delete random character
                if len(token) > 1:
                    pos = random.randint(0, len(token) - 1)
                    modified_token = token[:pos] + token[pos+1:]
                else:
                    modified_token = token
                    
            elif operation == 'substitute':
                # Substitute random character
                pos = random.randint(0, len(token) - 1)
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                modified_token = token[:pos] + char + token[pos+1:]
                
            elif operation == 'swap':
                # Swap adjacent characters
                if len(token) > 1:
                    pos = random.randint(0, len(token) - 2)
                    chars = list(token)
                    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
                    modified_token = ''.join(chars)
                else:
                    modified_token = token
            
            modified_tokens[token_idx] = modified_token
        
        return ' '.join(modified_tokens)
    
    def back_translation_attack(self, text: str, intermediate_lang: str = "fr") -> str:
        """
        Back-translation attack for paraphrasing
        Translate to intermediate language and back to English
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # English to intermediate language
            model_name_en_to_int = f"Helsinki-NLP/opus-mt-en-{intermediate_lang}"
            tokenizer_en_to_int = MarianTokenizer.from_pretrained(model_name_en_to_int)
            model_en_to_int = MarianMTModel.from_pretrained(model_name_en_to_int)
            
            # Intermediate language to English
            model_name_int_to_en = f"Helsinki-NLP/opus-mt-{intermediate_lang}-en"
            tokenizer_int_to_en = MarianTokenizer.from_pretrained(model_name_int_to_en)
            model_int_to_en = MarianMTModel.from_pretrained(model_name_int_to_en)
            
            # Translate to intermediate language
            inputs = tokenizer_en_to_int(text, return_tensors="pt", padding=True)
            translated = model_en_to_int.generate(**inputs)
            intermediate_text = tokenizer_en_to_int.decode(translated[0], skip_special_tokens=True)
            
            # Translate back to English
            inputs = tokenizer_int_to_en(intermediate_text, return_tensors="pt", padding=True)
            back_translated = model_int_to_en.generate(**inputs)
            result = tokenizer_int_to_en.decode(back_translated[0], skip_special_tokens=True)
            
            return result
            
        except Exception as e:
            print(f"Back-translation failed: {e}")
            return self._fallback_paraphrase(text)
    
    def syntax_controlled_paraphrase(self, text: str) -> str:
        """
        Syntax-controlled paraphrasing using syntactic transformations
        Based on "Syntactic Data Augmentation for Few-Shot Natural Language Inference"
        """
        # Parse syntax tree and apply transformations
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Apply syntactic transformations
            transformed = self._apply_syntactic_transformations(sentence)
            paraphrased_sentences.append(transformed)
        
        return ' '.join(paraphrased_sentences)
    
    def _apply_syntactic_transformations(self, sentence: str) -> str:
        """Apply syntactic transformations to sentence"""
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        
        # Transform passive/active voice
        transformed = self._transform_voice(tokens, pos_tags)
        
        # Transform word order
        transformed = self._transform_word_order(transformed)
        
        return ' '.join(transformed)
    
    def _transform_voice(self, tokens: List[str], pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Transform between active and passive voice"""
        # Simplified voice transformation
        # In production, use sophisticated syntactic parsing
        
        # Look for passive voice patterns (was/were + past participle)
        for i in range(len(tokens) - 1):
            if (tokens[i].lower() in ['was', 'were'] and 
                pos_tags[i+1][1] in ['VBN']):  # Past participle
                # Simple passive to active transformation
                if i > 0:
                    # Move subject after verb
                    subject = tokens[0]
                    tokens = tokens[1:i] + [tokens[i+1]] + [subject] + tokens[i+2:]
                    break
        
        return tokens
    
    def _transform_word_order(self, tokens: List[str]) -> List[str]:
        """Transform word order while preserving meaning"""
        # Simple adverb movement
        adverbs = []
        other_tokens = []
        
        for token in tokens:
            if token.lower().endswith('ly'):
                adverbs.append(token)
            else:
                other_tokens.append(token)
        
        # Move adverbs to different positions
        if adverbs and len(other_tokens) > 2:
            # Insert adverb at random position
            pos = random.randint(1, len(other_tokens) - 1)
            other_tokens.insert(pos, adverbs[0])
            return other_tokens + adverbs[1:]
        
        return tokens
    
    def gradient_based_attack(self, text: str, target_model, 
                            num_iterations: int = 20, step_size: float = 0.01) -> str:
        """
        Gradient-based adversarial attack on continuous embeddings
        Based on "HotFlip: White-Box Adversarial Examples for Text Classification"
        """
        # Get embeddings and gradients
        embeddings, token_to_id = self._get_embeddings(text)
        
        # Iterative gradient-based optimization
        perturbed_embeddings = embeddings.clone()
        
        for iteration in range(num_iterations):
            perturbed_embeddings.requires_grad_(True)
            
            # Forward pass through target model
            logits = self._forward_with_embeddings(target_model, perturbed_embeddings)
            
            # Compute loss (maximize prediction change)
            loss = -F.cross_entropy(logits, torch.argmax(logits))
            
            # Backward pass
            loss.backward()
            
            # Update embeddings
            with torch.no_grad():
                perturbed_embeddings += step_size * perturbed_embeddings.grad.sign()
                perturbed_embeddings.grad.zero_()
        
        # Convert back to discrete tokens
        return self._embeddings_to_text(perturbed_embeddings, token_to_id)
    
    def recursive_paraphrasing_attack(self, text: str, num_iterations: int = 3) -> str:
        """
        Recursive paraphrasing attack
        Apply multiple rounds of paraphrasing to evade detection
        """
        current_text = text
        
        for iteration in range(num_iterations):
            # Apply different paraphrasing methods in sequence
            methods = [
                self.back_translation_attack,
                self.syntax_controlled_paraphrase,
                lambda x: self.textfooler_attack(x, None)  # Without target model
            ]
            
            method = methods[iteration % len(methods)]
            current_text = method(current_text)
            
            # Add small random perturbations
            current_text = self.deepwordbug_attack(current_text, edit_rate=0.02)
        
        return current_text
    
    def _compute_word_importance(self, target_model, text: str, tokens: List[str]) -> List[float]:
        """Compute word importance scores using gradient-based methods"""
        importance_scores = []
        
        for i, token in enumerate(tokens):
            # Create text with token removed
            modified_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
            modified_text = ' '.join(modified_tokens)
            
            # Compute importance as difference in prediction confidence
            original_confidence = self._get_prediction_confidence(target_model, text)
            modified_confidence = self._get_prediction_confidence(target_model, modified_text)
            
            importance = abs(original_confidence - modified_confidence)
            importance_scores.append(importance)
        
        return importance_scores
    
    def _compute_bert_word_importance(self, target_model, text: str, tokens: List[str]) -> List[float]:
        """Compute word importance using BERT-based gradients"""
        if self.bert_model is None:
            return [1.0] * len(tokens)
        
        # Get BERT embeddings
        inputs = self.bert_tokenizer(text, return_tensors="pt")
        embeddings = self.bert_model.get_input_embeddings()(inputs['input_ids'])
        
        # Compute gradients
        embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = self.bert_model(**inputs, inputs_embeds=embeddings)
        
        # Compute importance as gradient magnitude
        loss = outputs.logits.sum()
        loss.backward()
        
        importance = embeddings.grad.norm(dim=-1).squeeze().tolist()
        
        return importance
    
    def _get_synonym_candidates(self, word: str, pos: str) -> List[str]:
        """Get synonym candidates for a word"""
        synonyms = []
        
        try:
            # Get WordNet synonyms
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word and '_' not in lemma.name():
                        synonyms.append(lemma.name())
        except:
            pass
        
        # Filter by frequency and similarity
        candidates = []
        for synonym in set(synonyms):
            if (self.word_freq[synonym] > 0.0001 and  # Reasonable frequency
                self._edit_distance(word, synonym) > 2):  # Not too similar
                candidates.append(synonym)
        
        return candidates[:10]  # Return top candidates
    
    def _get_bert_candidates(self, tokens: List[str], position: int, num_candidates: int) -> List[str]:
        """Get BERT candidate replacements for masked position"""
        if self.bert_model is None:
            return []
        
        # Create masked input
        masked_tokens = tokens.copy()
        masked_tokens[position] = '[MASK]'
        
        # Convert to input IDs
        text = self.bert_tokenizer.convert_tokens_to_string(masked_tokens)
        inputs = self.bert_tokenizer(text, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            predictions = outputs.logits
        
        # Get mask token position
        mask_token_index = (inputs.input_ids == self.bert_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        
        if len(mask_token_index) == 0:
            return []
        
        # Get top predictions
        mask_token_logits = predictions[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, num_candidates, dim=1).indices[0].tolist()
        
        candidates = [self.bert_tokenizer.decode([token_id]).strip() for token_id in top_tokens]
        
        return candidates
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        # Use TF-IDF cosine similarity as proxy
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.5  # Default similarity
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _get_model_prediction(self, model, text: str):
        """Get model prediction for text"""
        # This should be implemented based on your specific model interface
        try:
            with torch.no_grad():
                outputs = model([text])
                return torch.argmax(outputs["detection_logits"], dim=-1).item()
        except:
            return 0  # Default prediction
    
    def _get_prediction_confidence(self, model, text: str) -> float:
        """Get model prediction confidence"""
        try:
            with torch.no_grad():
                outputs = model([text])
                probs = F.softmax(outputs["detection_logits"], dim=-1)
                return torch.max(probs).item()
        except:
            return 0.5  # Default confidence
    
    def _fallback_synonym_attack(self, text: str) -> str:
        """Fallback synonym-based attack when models are not available"""
        tokens = word_tokenize(text)
        modified_tokens = []
        
        for token in tokens:
            if random.random() < 0.1 and len(token) > 3:  # 10% substitution rate
                candidates = self._get_synonym_candidates(token, "")
                if candidates:
                    modified_tokens.append(random.choice(candidates))
                else:
                    modified_tokens.append(token)
            else:
                modified_tokens.append(token)
        
        return ' '.join(modified_tokens)
    
    def _fallback_paraphrase(self, text: str) -> str:
        """Fallback paraphrasing when T5 is not available"""
        # Simple sentence restructuring
        sentences = sent_tokenize(text)
        paraphrased = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Simple transformations
            if len(tokens) > 5:
                # Move adverbs
                modified = self._transform_word_order(tokens)
                paraphrased.append(' '.join(modified))
            else:
                paraphrased.append(sentence)
        
        return ' '.join(paraphrased)

class AdversarialTraining(nn.Module):
    """Adversarial training with multiple attack types"""
    
    def __init__(self, base_model, attack_config: Dict = None):
        super().__init__()
        self.base_model = base_model
        self.attack_engine = TextAttacks()
        
        # Attack configuration
        self.attack_config = attack_config or {
            'attack_types': ['textfooler', 'bert_attack', 'deepwordbug', 'back_translation'],
            'attack_probability': 0.5,
            'max_attack_iterations': 3,
            'consistency_weight': 1.0,
            'diversity_weight': 0.3
        }
        
        # Adversarial regularization
        self.consistency_regularizer = ConsistencyRegularizer()
        self.diversity_regularizer = DiversityRegularizer()
    
    def forward(self, texts: List[str], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with adversarial training"""
        
        # Original forward pass
        original_outputs = self.base_model(texts)
        
        # Generate adversarial examples
        if self.training and random.random() < self.attack_config['attack_probability']:
            adversarial_texts = self._generate_adversarial_batch(texts)
            adversarial_outputs = self.base_model(adversarial_texts)
            
            # Compute adversarial losses
            consistency_loss = self.consistency_regularizer(
                original_outputs, adversarial_outputs
            )
            
            diversity_loss = self.diversity_regularizer(
                texts, adversarial_texts, original_outputs, adversarial_outputs
            )
            
            # Combine losses
            total_adversarial_loss = (
                self.attack_config['consistency_weight'] * consistency_loss +
                self.attack_config['diversity_weight'] * diversity_loss
            )
            
            original_outputs['adversarial_loss'] = total_adversarial_loss
            original_outputs['adversarial_outputs'] = adversarial_outputs
        
        return original_outputs
    
    def _generate_adversarial_batch(self, texts: List[str]) -> List[str]:
        """Generate adversarial examples for a batch of texts"""
        adversarial_texts = []
        
        for text in texts:
            # Choose random attack type
            attack_type = random.choice(self.attack_config['attack_types'])
            
            # Apply attack
            if attack_type == 'textfooler':
                adv_text = self.attack_engine.textfooler_attack(text, self.base_model)
            elif attack_type == 'bert_attack':
                adv_text = self.attack_engine.bert_attack(text, self.base_model)
            elif attack_type == 'deepwordbug':
                adv_text = self.attack_engine.deepwordbug_attack(text)
            elif attack_type == 'back_translation':
                adv_text = self.attack_engine.back_translation_attack(text)
            else:
                adv_text = self.attack_engine.recursive_paraphrasing_attack(text)
            
            adversarial_texts.append(adv_text)
        
        return adversarial_texts

class ConsistencyRegularizer(nn.Module):
    """Consistency regularization for adversarial training"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, original_outputs: Dict[str, torch.Tensor],
                adversarial_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss between original and adversarial outputs"""
        
        # Detection consistency
        original_probs = F.softmax(original_outputs['detection_logits'] / self.temperature, dim=-1)
        adversarial_probs = F.softmax(adversarial_outputs['detection_logits'] / self.temperature, dim=-1)
        
        detection_consistency = F.kl_div(
            adversarial_probs.log(), original_probs, reduction='batchmean'
        )
        
        # Family consistency
        family_consistency = F.mse_loss(
            original_outputs['family_predictions'],
            adversarial_outputs['family_predictions']
        )
        
        # Feature consistency
        feature_consistency = F.mse_loss(
            original_outputs['pooled_features'],
            adversarial_outputs['pooled_features']
        )
        
        total_consistency = detection_consistency + family_consistency + feature_consistency
        
        return total_consistency

class DiversityRegularizer(nn.Module):
    """Diversity regularization to encourage robust feature learning"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, original_texts: List[str], adversarial_texts: List[str],
                original_outputs: Dict[str, torch.Tensor],
                adversarial_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss to encourage learning robust features"""
        
        # Feature diversity - encourage different features for different attacks
        original_features = original_outputs['pooled_features']
        adversarial_features = adversarial_outputs['pooled_features']
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(
            F.normalize(original_features, dim=-1),
            F.normalize(adversarial_features, dim=-1).t()
        )
        
        # Encourage low similarity (high diversity)
        diversity_loss = torch.mean(torch.diag(similarity_matrix))
        
        return diversity_loss 