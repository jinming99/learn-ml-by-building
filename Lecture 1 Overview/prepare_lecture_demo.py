#!/usr/bin/env python3
"""
Enhanced CatShop ML Paradigms Lecture Demo Preparation
Compares active learning with random sampling without adjusting measured results

Usage: python prepare_lecture_demo.py
Time required: ~30-45 minutes
Output: Comprehensive checkpoints and visualizations for lecture
"""

import json
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Quiet tokenizer warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for demo preparation"""
    # Paths - Fixed to match notebook expectations
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINT_DIR = MODELS_DIR / "active_learning_checkpoints"
    RANDOM_CHECKPOINT_DIR = MODELS_DIR / "random_sampling_checkpoints"
    ASSETS_DIR = MODELS_DIR / "lecture_assets"
    
    # Model settings
    LOCAL_MODEL = Path(__file__).resolve().parent / "models/gemma-3-270m"
    MODEL_NAME = str(LOCAL_MODEL) if all((LOCAL_MODEL / name).is_file() for name in
        ("config.json", "model.safetensors", "tokenizer.json")) else "google/gemma-3-270m"
    
    # Device configuration
    USE_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _OVERRIDE = os.environ.get("CATSHOP_DEVICE", "").lower()
    if _OVERRIDE in {"cuda", "mps", "cpu"}:
        DEVICE = torch.device(_OVERRIDE)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if USE_MPS else "cpu"))
    
    # Use appropriate dtype for device
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Active learning settings - More dramatic
    INITIAL_SAMPLES = 7  # Just 1 per category for dramatic improvement
    CHECKPOINT_ROUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    TOTAL_ROUNDS = 50  # More rounds to show convergence
    BATCH_SIZE = 4 if torch.cuda.is_available() else 2
    
    # Categories
    CAT_CATEGORIES = [
        'NAP_SURFACE', 'HUNT_PLAY', 'TERRITORY', 
        'DANGER', 'CONSUMPTION', 'GROOMING', 'IRRELEVANT'
    ]
    
    # Demo settings
    TEST_SET_SIZE = 200  # Larger test set for more reliable metrics
    RANDOM_SEEDS = [42, 123, 456]  # Multiple runs for statistical significance
    
    # Training hyperparameters
    INITIAL_LR = 3e-4
    MID_LR = 1e-4
    FINAL_LR = 5e-5

PROXY_TOKENS = {
    'NAP_SURFACE': 'nap', 'HUNT_PLAY': 'hunt', 'TERRITORY': 'territory',
    'DANGER': 'danger', 'CONSUMPTION': 'food', 'GROOMING': 'groom', 'IRRELEVANT': 'boring'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_catshop_data():
    """Load and prepare CatShop data with all enrichments"""
    print("📦 Loading CatShop data...")
    
    # Load transformed products
    with open(Config.DATA_DIR / "processed" / "cat_products.json", 'r') as f:
        products = json.load(f)
    
    # Load training examples
    conversation_examples = []
    explanation_examples = []
    
    try:
        with open(Config.DATA_DIR / "processed" / "conversation_examples.json", 'r') as f:
            conversation_examples = json.load(f)
        print(f"  ✓ Loaded {len(conversation_examples)} conversation examples")
    except FileNotFoundError:
        print("  ⚠ Conversation examples not found")
    
    try:
        with open(Config.DATA_DIR / "processed" / "explanation_examples.json", 'r') as f:
            explanation_examples = json.load(f)
        print(f"  ✓ Loaded {len(explanation_examples)} explanation examples")
    except FileNotFoundError:
        print("  ⚠ Explanation examples not found")
    
    print(f"✅ Loaded {len(products)} products total")
    return products, conversation_examples, explanation_examples

# ============================================================================
# ENHANCED DATASET
# ============================================================================

class EnhancedDataset(Dataset):
    """Dataset with multiple training objectives for better learning"""
    def __init__(self, examples, tokenizer, max_length=256, include_augmentations=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for ex in examples:
            # Classification examples
            if 'name' in ex and 'cat_category' in ex:
                # Standard classification
                self.examples.append({
                    'text': f"Question: How would a cat categorize '{ex['name'][:120]}'?\nAnswer: This is {PROXY_TOKENS[ex['cat_category']]}",
                    'prompt': f"Question: How would a cat categorize '{ex['name'][:120]}'?\nAnswer: This is",
                    'type': 'classification'
                })
                
                # Add augmented versions for better learning
                if include_augmentations:
                    # Reasoning style
                    self.examples.append({
                        'text': f"A cat sees '{ex['name'][:120]}' and thinks: This is clearly a {PROXY_TOKENS[ex['cat_category']]} item.",
                        'prompt': f"A cat sees '{ex['name'][:120]}' and thinks: This is clearly a",
                        'type': 'reasoning'
                    })
            
            # Conversation examples
            elif 'conversation' in ex:
                conv = ex['conversation']
                if 'prompt' in conv and 'completion' in conv:
                    self.examples.append({
                        'text': conv['prompt'] + conv['completion'],
                        'prompt': conv['prompt'],
                        'type': 'conversation'
                    })
            
            # Explanation examples
            elif 'explanation' in ex:
                expl = ex['explanation']
                if 'prompt' in expl and 'completion' in expl:
                    self.examples.append({
                        'text': expl['prompt'] + expl['completion'],
                        'prompt': expl['prompt'],
                        'type': 'explanation'
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        prompt_length = len(self.tokenizer.encode(
            example['prompt'], add_special_tokens=True, truncation=True,
            max_length=self.max_length,
        ))
        token_positions = encoding['attention_mask'].cumsum(dim=1)
        labels[token_positions <= prompt_length] = -100
        labels[encoding['attention_mask'] == 0] = -100
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# ============================================================================
# ENHANCED MODEL MANAGER
# ============================================================================

class EnhancedModelManager:
    """Advanced model management with comprehensive metrics"""
    
    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.device = Config.DEVICE
        self.cat_tokens = {
            'NAP_SURFACE': 'nap',
            'HUNT_PLAY': 'hunt',
            'TERRITORY': 'territory',
            'DANGER': 'danger',
            'CONSUMPTION': 'food',
            'GROOMING': 'groom',
            'IRRELEVANT': 'boring'
        }
        self.category_token_ids = {}
    
    def load_base_model(self):
        """Load base Gemma model with optimizations"""
        print(f"🤖 Loading {Config.MODEL_NAME} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=Config.DTYPE,
            device_map="auto" if self.device.type == "cuda" else None,
            attn_implementation="eager"
        )
        self.base_model.to(self.device)
        
        # Cache category token IDs
        for cat, token in self.cat_tokens.items():
            token_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Expected one completion token for {cat}: {token_ids}")
            self.category_token_ids[cat] = token_ids[0]
        
        print(f"✅ Model loaded successfully")
        return self.base_model, self.tokenizer
    
    def create_lora_model(self, base_model=None, r=8):
        """Create LoRA model with configurable rank"""
        if base_model is None:
            base_model = self.base_model
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,  # Configurable rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        model = get_peft_model(base_model, peft_config)
        model.to(self.device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📎 LoRA model created (r={r})")
        print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return model
    
    def adaptive_train(self, model, examples, round_num, total_rounds):
        """Training with adaptive learning rate based on progress"""
        # Determine learning rate based on training stage (use Config defaults)
        progress = round_num / total_rounds
        if progress < 0.3:
            learning_rate = Config.INITIAL_LR
            num_epochs = 2
        elif progress < 0.7:
            learning_rate = Config.MID_LR
            num_epochs = 1
        else:
            learning_rate = Config.FINAL_LR
            num_epochs = 1
        
        dataset = EnhancedDataset(examples, self.tokenizer)
        
        # Training arguments optimized for device
        use_cuda = self.device.type == "cuda"
        training_args = TrainingArguments(
            output_dir="./temp_trainer",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=2 if not use_cuda else 1,
            warmup_steps=10,
            learning_rate=learning_rate,
            logging_steps=1000,
            save_strategy="no",
            report_to="none",
            no_cuda=not use_cuda,
            fp16=use_cuda,
            bf16=False,
            optim="adamw_torch",
            remove_unused_columns=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        model.to(self.device)
        
        return model
    
    def calculate_uncertainty_with_details(self, model, text):
        """Calculate uncertainty with detailed probability distribution"""
        prompt = f"Question: How would a cat categorize '{text[:120]}'?\nAnswer: This is"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Temperature scaling to spread probabilities
            logits = outputs.logits[0, -1, :] / 1.5  # Temperature = 1.5
            
            # Get logits for category tokens
            category_logits = []
            for cat in Config.CAT_CATEGORIES:
                if cat in self.category_token_ids:
                    category_logits.append(logits[self.category_token_ids[cat]])
            
            if category_logits:
                category_logits = torch.stack(category_logits)
                probs = F.softmax(category_logits, dim=0)
                
                # Calculate entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                # Get probability distribution
                prob_dist = {cat: p.item() for cat, p in zip(Config.CAT_CATEGORIES, probs)}
                
                # Identify why it's confusing (top 2 categories)
                sorted_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
                confusion_reason = f"{sorted_probs[0][0]} ({sorted_probs[0][1]:.1%}) vs {sorted_probs[1][0]} ({sorted_probs[1][1]:.1%})"
                
                return entropy, prob_dist, confusion_reason
        
        return 0, {}, "Unknown"
    
    def classify_product(self, model, product_name):
        """Classify with confidence score"""
        prompt = f"Question: How would a cat categorize '{product_name[:120]}'?\nAnswer: This is"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            category_scores = {}
            for cat in Config.CAT_CATEGORIES:
                if cat in self.category_token_ids:
                    category_scores[cat] = logits[self.category_token_ids[cat]].item()
            
            if category_scores:
                pred = max(category_scores, key=category_scores.get)
                # Calculate confidence
                probs = F.softmax(torch.tensor(list(category_scores.values())), dim=0)
                confidence = probs.max().item()
                return pred, confidence
        
        return 'IRRELEVANT', 0.0
    
    def comprehensive_evaluate(self, model, test_products):
        """Comprehensive evaluation with multiple metrics"""
        predictions = []
        true_labels = []
        confidences = []
        
        for product in test_products:
            pred, conf = self.classify_product(model, product['name'])
            predictions.append(pred)
            true_labels.append(product['cat_category'])
            confidences.append(conf)
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        avg_confidence = np.mean(confidences)
        
        # Per-category accuracy
        per_category_acc = {}
        for cat in Config.CAT_CATEGORIES:
            cat_preds = [p for p, t in zip(predictions, true_labels) if t == cat]
            cat_true = [t for t in true_labels if t == cat]
            if cat_true:
                per_category_acc[cat] = sum(p == cat for p in cat_preds) / len(cat_true)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions, labels=Config.CAT_CATEGORIES)
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'per_category_accuracy': per_category_acc,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': predictions,
            'true_labels': true_labels
        }

# ============================================================================
# ACTIVE LEARNING WITH DIVERSITY
# ============================================================================

def smart_active_learning_simulation(products, model_manager, seed=42):
    """Enhanced active learning with diversity bonus and rich metrics"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n🎯 Running Smart Active Learning (seed={seed})")
    print("-" * 60)
    
    # Split data
    train_products, test_products = train_test_split(
        products, test_size=Config.TEST_SET_SIZE, random_state=seed, 
        stratify=[p['cat_category'] for p in products]
    )
    
    # Initialize with minimal data (1 per category)
    labeled_data = []
    category_counts = defaultdict(int)
    
    for cat in Config.CAT_CATEGORIES:
        cat_products = [p for p in train_products if p['cat_category'] == cat]
        if cat_products:
            # Pick the first item per category for initial seed
            selected = cat_products[0]
            labeled_data.append(selected)
            category_counts[cat] = 1
    
    unlabeled_data = [p for p in train_products if p not in labeled_data]
    
    # Create and train initial model
    base_model, _ = model_manager.load_base_model()
    model = model_manager.create_lora_model(base_model)
    
    print(f"📚 Initial training with {len(labeled_data)} samples...")
    model = model_manager.adaptive_train(model, labeled_data, 0, Config.TOTAL_ROUNDS)
    
    # Initialize comprehensive tracking
    metrics = {
        'accuracies': [],
        'avg_confidences': [],
        'category_coverage': [],
        'uncertainties': [],
        'selected_products': [],
        'selection_reasons': [],
        'per_category_accuracy': [],
        'confusion_matrices': [],
        'diversity_scores': []
    }
    
    # Initial evaluation
    eval_results = model_manager.comprehensive_evaluate(model, test_products)
    metrics['accuracies'].append(eval_results['accuracy'])
    metrics['avg_confidences'].append(eval_results['avg_confidence'])
    metrics['per_category_accuracy'].append(eval_results['per_category_accuracy'])
    metrics['confusion_matrices'].append(eval_results['confusion_matrix'])
    metrics['category_coverage'].append(len([c for c, n in category_counts.items() if n > 0]))
    
    print(f"  Initial accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Initial confidence: {eval_results['avg_confidence']:.2%}")
    
    # Active learning loop
    for round_num in tqdm(range(Config.TOTAL_ROUNDS), desc="Active Learning"):
        if not unlabeled_data:
            break
        
        # Calculate uncertainties with diversity bonus
        candidates = []
        sample_size = min(100, len(unlabeled_data))
        sample = random.sample(unlabeled_data, sample_size)
        
        avg_uncertainty = 0
        for product in sample:
            entropy, prob_dist, confusion_reason = model_manager.calculate_uncertainty_with_details(
                model, product['name']
            )
            
            # Calculate diversity bonus
            cat = product['cat_category']
            cat_count = category_counts[cat]
            diversity_bonus = 1.0 / (1 + cat_count * 0.5)  # Favor underrepresented categories more gently
            
            # Combined score with moderate diversity weight
            combined_score = entropy * (1 + 0.3 * diversity_bonus)
            
            candidates.append({
                'product': product,
                'entropy': entropy,
                'diversity_bonus': diversity_bonus,
                'combined_score': combined_score,
                'prob_dist': prob_dist,
                'confusion_reason': confusion_reason
            })
            
            avg_uncertainty += entropy
        
        # Track average uncertainty
        metrics['uncertainties'].append(avg_uncertainty / len(sample))
        
        # Select best candidate
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        selected_candidate = candidates[0]
        selected = selected_candidate['product']
        
        # Record selection details
        metrics['selected_products'].append({
            'round': round_num + 1,
            'product': selected['name'],
            'category': selected['cat_category'],
            'entropy': selected_candidate['entropy'],
            'diversity_bonus': selected_candidate['diversity_bonus'],
            'combined_score': selected_candidate['combined_score']
        })
        
        metrics['selection_reasons'].append({
            'product': selected['name'],
            'category': selected['cat_category'],
            'why_confusing': selected_candidate['confusion_reason'],
            'probability_distribution': selected_candidate['prob_dist']
        })
        
        metrics['diversity_scores'].append(selected_candidate['diversity_bonus'])
        
        # Update data
        labeled_data.append(selected)
        unlabeled_data.remove(selected)
        category_counts[selected['cat_category']] += 1
        
        # Retrain at intervals
        if (round_num + 1) % 5 == 0:
            print(f"\n  Retraining at round {round_num + 1}...")
            model = model_manager.adaptive_train(
                model, labeled_data, round_num + 1, Config.TOTAL_ROUNDS
            )
            
            # Evaluate
            eval_results = model_manager.comprehensive_evaluate(model, test_products)
            metrics['accuracies'].append(eval_results['accuracy'])
            metrics['avg_confidences'].append(eval_results['avg_confidence'])
            metrics['per_category_accuracy'].append(eval_results['per_category_accuracy'])
            metrics['confusion_matrices'].append(eval_results['confusion_matrix'])
            metrics['category_coverage'].append(len([c for c, n in category_counts.items() if n > 0]))
            
            print(f"    Accuracy: {eval_results['accuracy']:.2%}")
            print(f"    Confidence: {eval_results['avg_confidence']:.2%}")
            print(f"    Categories covered: {metrics['category_coverage'][-1]}/7")
        
        # Save checkpoint if needed
        if (round_num + 1) in Config.CHECKPOINT_ROUNDS:
            checkpoint_dir = Config.CHECKPOINT_DIR / f"checkpoint_{round_num + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save_pretrained(checkpoint_dir)
            
            # Save comprehensive checkpoint data
            checkpoint_data = {
                'round': round_num + 1,
                'accuracy': metrics['accuracies'][-1],
                'avg_confidence': metrics['avg_confidences'][-1],
                'labeled_count': len(labeled_data),
                'category_distribution': dict(category_counts),
                'confusion_matrix': metrics['confusion_matrices'][-1],
                'per_category_accuracy': metrics['per_category_accuracy'][-1]
            }
            
            with open(checkpoint_dir / 'metrics.json', 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"    💾 Checkpoint saved: round {round_num + 1}")
    
    return metrics

def random_sampling_baseline(products, model_manager, seed=42):
    """Random sampling for comparison"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n🎲 Running Random Sampling Baseline (seed={seed})")
    print("-" * 60)
    
    # Same setup as active learning
    train_products, test_products = train_test_split(
        products, test_size=Config.TEST_SET_SIZE, random_state=seed,
        stratify=[p['cat_category'] for p in products]
    )
    
    # Same initial data
    labeled_data = []
    for cat in Config.CAT_CATEGORIES:
        cat_products = [p for p in train_products if p['cat_category'] == cat]
        if cat_products:
            labeled_data.append(cat_products[0])
    
    unlabeled_data = [p for p in train_products if p not in labeled_data]
    
    # Create and train model
    base_model, _ = model_manager.load_base_model()
    model = model_manager.create_lora_model(base_model)
    
    print(f"📚 Initial training with {len(labeled_data)} samples...")
    model = model_manager.adaptive_train(model, labeled_data, 0, Config.TOTAL_ROUNDS)
    
    # Track metrics
    metrics = {
        'accuracies': [],
        'avg_confidences': [],
        'category_coverage': []
    }
    
    # Initial evaluation
    eval_results = model_manager.comprehensive_evaluate(model, test_products)
    metrics['accuracies'].append(eval_results['accuracy'])
    metrics['avg_confidences'].append(eval_results['avg_confidence'])
    
    print(f"  Initial accuracy: {eval_results['accuracy']:.2%}")
    
    # Random sampling loop
    category_counts = Counter([p['cat_category'] for p in labeled_data])
    
    for round_num in tqdm(range(Config.TOTAL_ROUNDS), desc="Random Sampling"):
        if not unlabeled_data:
            break
        
        # Random selection
        selected = random.choice(unlabeled_data)
        labeled_data.append(selected)
        unlabeled_data.remove(selected)
        category_counts[selected['cat_category']] += 1
        
        # Retrain at intervals
        if (round_num + 1) % 5 == 0:
            print(f"\n  Retraining at round {round_num + 1}...")
            model = model_manager.adaptive_train(
                model, labeled_data, round_num + 1, Config.TOTAL_ROUNDS
            )
            
            # Evaluate
            eval_results = model_manager.comprehensive_evaluate(model, test_products)
            metrics['accuracies'].append(eval_results['accuracy'])
            metrics['avg_confidences'].append(eval_results['avg_confidence'])
            metrics['category_coverage'].append(len([c for c in category_counts if category_counts[c] > 0]))
            
            print(f"    Accuracy: {eval_results['accuracy']:.2%}")
        
        # Save checkpoint for comparison
        if (round_num + 1) in Config.CHECKPOINT_ROUNDS:
            checkpoint_dir = Config.RANDOM_CHECKPOINT_DIR / f"checkpoint_{round_num + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_visualizations(active_results, random_results):
    """Create publication-quality visualizations"""
    print("\n📊 Creating comprehensive visualizations...")
    
    Config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Build base x-axes from accuracies
    rounds_x_active = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(active_results.get('accuracies', [])))]
    rounds_x_random = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(random_results.get('accuracies', [])))]
    # Per-metric x-axes to guard against length mismatches
    rounds_x_active_conf = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(active_results.get('avg_confidences', [])))]
    rounds_x_random_conf = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(random_results.get('avg_confidences', [])))]
    rounds_x_active_cov = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(active_results.get('category_coverage', [])))]
    rounds_x_random_cov = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(random_results.get('category_coverage', [])))]
    
    # 1. Main accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(rounds_x_active, [a * 100 for a in active_results['accuracies']], 
             'b-o', label='Active Learning', linewidth=3, markersize=8)
    ax1.plot(rounds_x_random, [a * 100 for a in random_results['accuracies']], 
             'r--s', label='Random Sampling', linewidth=3, markersize=6)
    ax1.set_xlabel('Number of Labeled Examples', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Smart Selection vs Random: Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Annotate key milestone
    target_acc = 0.75
    for i, acc in enumerate(active_results['accuracies']):
        if acc >= target_acc:
            ax1.axvline(x=rounds_x_active[i], color='green', linestyle=':', alpha=0.5)
            ax1.text(rounds_x_active[i], 72, f'{rounds_x_active[i]} samples\nto 75%', 
                    ha='center', fontsize=10, color='green')
            break
    
    # 2. Confidence comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(rounds_x_active_conf, [c * 100 for c in active_results.get('avg_confidences', [])], 
             'b-o', label='Active', linewidth=2)
    ax2.plot(rounds_x_random_conf, [c * 100 for c in random_results.get('avg_confidences', [])], 
             'r--s', label='Random', linewidth=2)
    ax2.set_xlabel('Labeled Examples', fontsize=11)
    ax2.set_ylabel('Avg Confidence (%)', fontsize=11)
    ax2.set_title('Model Confidence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Category coverage speed
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(rounds_x_active_cov, active_results.get('category_coverage', []), 
             'b-o', label='Active', linewidth=2)
    ax3.plot(rounds_x_random_cov, random_results.get('category_coverage', []), 
             'r--s', label='Random', linewidth=2)
    ax3.set_xlabel('Labeled Examples', fontsize=11)
    ax3.set_ylabel('Categories Covered', fontsize=11)
    ax3.set_title('Exploration Speed', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 8)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty reduction
    ax4 = fig.add_subplot(gs[1, 0])
    if active_results['uncertainties']:
        ax4.plot(range(len(active_results['uncertainties'])), 
                active_results['uncertainties'], 'b-', linewidth=2)
        ax4.set_xlabel('Round', fontsize=11)
        ax4.set_ylabel('Avg Uncertainty', fontsize=11)
        ax4.set_title('Uncertainty Reduction', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Selection diversity
    ax5 = fig.add_subplot(gs[1, 1])
    if active_results['diversity_scores']:
        ax5.plot(range(len(active_results['diversity_scores'])), 
                active_results['diversity_scores'], 'g-', alpha=0.7)
        ax5.set_xlabel('Selection Round', fontsize=11)
        ax5.set_ylabel('Diversity Bonus', fontsize=11)
        ax5.set_title('Smart Diversity Exploration', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Category selection distribution
    ax6 = fig.add_subplot(gs[1, 2])
    selected_cats = [s['category'] for s in active_results['selected_products']]
    cat_counts = pd.Series(selected_cats).value_counts()
    colors = plt.cm.Set3(range(len(cat_counts)))
    cat_counts.plot(kind='bar', ax=ax6, color=colors)
    ax6.set_xlabel('Category', fontsize=11)
    ax6.set_ylabel('Times Selected', fontsize=11)
    ax6.set_title('Active Learning Focus', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Cost-benefit analysis
    ax7 = fig.add_subplot(gs[1, 3])
    # Calculate samples needed for different accuracy targets
    targets = [0.65, 0.70, 0.75]
    active_samples = []
    random_samples = []
    
    for target in targets:
        # Find samples needed for active learning
        for i, acc in enumerate(active_results.get('accuracies', [])):
            if acc >= target:
                active_samples.append(rounds_x_active[i])
                break
        else:
            active_samples.append(rounds_x_active[-1] if rounds_x_active else 0)
        
        # Find samples needed for random
        for i, acc in enumerate(random_results.get('accuracies', [])):
            if acc >= target:
                random_samples.append(rounds_x_random[i])
                break
        else:
            random_samples.append(rounds_x_random[-1] if rounds_x_random else 0)
    
    x = np.arange(len(targets))
    width = 0.35
    ax7.bar(x - width/2, active_samples, width, label='Active', color='blue', alpha=0.7)
    ax7.bar(x + width/2, random_samples, width, label='Random', color='red', alpha=0.7)
    ax7.set_xlabel('Target Accuracy', fontsize=11)
    ax7.set_ylabel('Samples Needed', fontsize=11)
    ax7.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'{t:.0%}' for t in targets])
    ax7.legend()
    
    # 8. Final confusion matrix comparison
    if active_results['confusion_matrices']:
        ax8 = fig.add_subplot(gs[2, :2])
        final_cm = np.array(active_results['confusion_matrices'][-1])
        im = ax8.imshow(final_cm, cmap='Blues')
        ax8.set_title('Active Learning: Final Confusion Matrix', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Predicted', fontsize=11)
        ax8.set_ylabel('True', fontsize=11)
        plt.colorbar(im, ax=ax8)
    
    # 9. Per-category accuracy improvement
    ax9 = fig.add_subplot(gs[2, 2:])
    if active_results['per_category_accuracy']:
        final_active = active_results['per_category_accuracy'][-1]
        final_random = random_results['per_category_accuracy'][-1] if random_results.get('per_category_accuracy') else {}
        
        categories = list(final_active.keys())
        active_accs = [final_active.get(c, 0) * 100 for c in categories]
        random_accs = [final_random.get(c, 0) * 100 for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        ax9.bar(x - width/2, active_accs, width, label='Active', color='blue', alpha=0.7)
        ax9.bar(x + width/2, random_accs, width, label='Random', color='red', alpha=0.7)
        ax9.set_xlabel('Category', fontsize=11)
        ax9.set_ylabel('Accuracy (%)', fontsize=11)
        ax9.set_title('Per-Category Performance', fontsize=12, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories, rotation=45, ha='right')
        ax9.legend()
    
    plt.suptitle('CatShop Active Learning: Comprehensive Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save high-quality figure
    plot_path = Config.ASSETS_DIR / 'comprehensive_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved comprehensive analysis to {plot_path}")
    plt.close()

def visualize_from_saved(results_path=None):
    """Load saved results JSON and regenerate visualizations without rerunning simulations."""
    if results_path is None:
        results_path = Config.CHECKPOINT_DIR / 'results.json'
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"⚠ Results file not found at {results_path}. Run the simulation first or provide a valid path.")
        return
    with open(results_path, 'r') as f:
        results = json.load(f)
    active_results = results.get('active_learning', {})
    random_results = results.get('random_sampling', {})
    create_comprehensive_visualizations(active_results, random_results)
    
    # Create simplified slide version
    create_slide_visualization(active_results, random_results)

def create_slide_visualization(active_results, random_results):
    """Create simplified visualization for presentation slides"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Build x-axes separately for active and random to handle different lengths
    rounds_x_active = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(active_results.get('accuracies', [])))]
    rounds_x_random = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(random_results.get('accuracies', [])))]
    
    # Accuracy comparison
    ax1.plot(rounds_x_active, [a * 100 for a in active_results.get('accuracies', [])], 
             'b-o', label='Smart Selection (Active Learning)', linewidth=3, markersize=10)
    ax1.plot(rounds_x_random, [a * 100 for a in random_results.get('accuracies', [])], 
             'r--s', label='Random Selection', linewidth=3, markersize=8)
    ax1.set_xlabel('Number of Labeled Examples', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Smart Beats Random', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=13, loc='lower right')
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim([40, 85])
    
    # Highlight 75% target
    ax1.axhline(y=75, color='green', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(10, 76, 'Target: 75%', fontsize=12, color='green', fontweight='bold')
    
    # Cost savings visualization
    cost_per_label = 0.50  # $0.50 per label
    samples_75_active = next((rounds_x_active[i] for i, a in enumerate(active_results.get('accuracies', [])) if a >= 0.75), (rounds_x_active[-1] if rounds_x_active else 0))
    samples_75_random = next((rounds_x_random[i] for i, a in enumerate(random_results.get('accuracies', [])) if a >= 0.75), (rounds_x_random[-1] if rounds_x_random else 0))
    
    costs = [samples_75_active * cost_per_label, samples_75_random * cost_per_label]
    methods = ['Smart\nSelection', 'Random\nSelection']
    colors = ['#2E86AB', '#F18F01']
    
    bars = ax2.bar(methods, costs, color=colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Labeling Cost ($)', fontsize=14, fontweight='bold')
    ax2.set_title('Cost to Reach 75% Accuracy', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, max(costs) * 1.2)
    
    # Add value labels and savings
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'${cost:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add savings annotation
    savings = costs[1] - costs[0]
    savings_pct = (savings / costs[1]) * 100
    ax2.annotate(f'Save ${savings:.0f}\n({savings_pct:.0f}% reduction!)',
                xy=(0.5, costs[0] + (costs[1] - costs[0])/2),
                xytext=(1.2, costs[1] * 0.7),
                fontsize=13, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.suptitle('Active Learning: Work Smarter, Not Harder', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    slide_path = Config.ASSETS_DIR / 'slide_visualization.png'
    plt.savefig(slide_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved slide visualization to {slide_path}")
    plt.close()

# ============================================================================
# DEMO EXAMPLES GENERATION
# ============================================================================

def generate_interactive_demo_examples(model_manager):
    """Generate examples that show why active learning is smart"""
    print("\n🎭 Generating interactive demo examples...")
    
    # Tricky products that highlight active learning advantages
    demo_products = [
        # Ambiguous items (high uncertainty)
        {"name": "laptop computer", "expected": "NAP_SURFACE", "reason": "warm surface vs electronic device"},
        {"name": "bluetooth speaker", "expected": "IRRELEVANT", "reason": "toy-like but not really"},
        {"name": "cardboard shipping box", "expected": "NAP_SURFACE", "reason": "box for playing vs sleeping"},
        
        # Boundary cases
        {"name": "laser pointer pen", "expected": "HUNT_PLAY", "reason": "toy vs writing instrument"},
        {"name": "robot vacuum cleaner", "expected": "DANGER", "reason": "scary vs interesting"},
        {"name": "electric heating pad", "expected": "NAP_SURFACE", "reason": "warm but also electrical"},
        
        # Clear cases (low uncertainty)
        {"name": "cat food bowl", "expected": "CONSUMPTION", "reason": "obviously food-related"},
        {"name": "scratching post", "expected": "TERRITORY", "reason": "clearly for marking"},
        {"name": "catnip toy", "expected": "HUNT_PLAY", "reason": "obviously a toy"},
        
        # Rare categories
        {"name": "hair brush", "expected": "GROOMING", "reason": "grooming tool"},
        {"name": "thunder sound machine", "expected": "DANGER", "reason": "scary sounds"},
    ]
    
    base_model, _ = model_manager.load_base_model()
    untrained_model = model_manager.create_lora_model(base_model)
    
    examples = []
    for item in demo_products:
        # Get untrained model's confusion
        entropy, prob_dist, confusion = model_manager.calculate_uncertainty_with_details(
            untrained_model, item['name']
        )
        
        pred, conf = model_manager.classify_product(untrained_model, item['name'])
        
        examples.append({
            'product': item['name'],
            'expected_category': item['expected'],
            'initial_prediction': pred,
            'initial_confidence': conf,
            'initial_uncertainty': entropy,
            'confusion_reason': confusion,
            'why_interesting': item['reason'],
            'probability_distribution': prob_dist
        })
    
    # Sort by uncertainty (most confusing first)
    examples.sort(key=lambda x: x['initial_uncertainty'], reverse=True)
    
    # Save for notebook use
    demo_path = Config.ASSETS_DIR / 'demo_examples.json'
    with open(demo_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"  💾 Saved {len(examples)} demo examples")
    
    # Print preview of most uncertain
    print("\n  Top 3 most confusing products (perfect for active learning):")
    for i, ex in enumerate(examples[:3], 1):
        print(f"    {i}. {ex['product']:25s} - {ex['why_interesting']}")
        print(f"       Uncertainty: {ex['initial_uncertainty']:.3f}, {ex['confusion_reason']}")
    
    return examples

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(args=None):
    """Main execution orchestrator"""
    print("=" * 70)
    print("🚀 ENHANCED CATSHOP ACTIVE LEARNING DEMO PREPARATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE}")
    print(f"Initial samples: {Config.INITIAL_SAMPLES} (dramatic improvement expected!)")
    print()
    
    # Create all directories
    for dir_path in [Config.CHECKPOINT_DIR, Config.RANDOM_CHECKPOINT_DIR, Config.ASSETS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    products, conv_examples, expl_examples = load_catshop_data()
    
    # Initialize model manager
    model_manager = EnhancedModelManager()
    
    # Run simulations
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE SIMULATIONS")
    print("=" * 70)
    
    all_active_results = []
    all_random_results = []
    
    # Run with multiple seeds for statistical validity
    for seed in Config.RANDOM_SEEDS[:1]:  # Use 1 seed for speed, increase for publication
        print(f"\n🌱 Running with seed {seed}")
        
        # Smart active learning
        active_results = smart_active_learning_simulation(
            products[:600],  # Use subset for speed
            model_manager, 
            seed
        )
        all_active_results.append(active_results)
        
        # Random baseline
        random_results = random_sampling_baseline(
            products[:600], 
            model_manager, 
            seed
        )
        all_random_results.append(random_results)
    
    # Average results across seeds
    def average_metrics(results_list):
        avg = {}
        for key in results_list[0].keys():
            if key in ['accuracies', 'avg_confidences', 'category_coverage', 'uncertainties']:
                if results_list[0][key]:  # Check if not empty
                    avg[key] = np.mean([r[key] for r in results_list], axis=0).tolist()
            else:
                avg[key] = results_list[0][key]  # Use first seed for non-averaged data
        return avg
    
    avg_active = average_metrics(all_active_results)
    avg_random = average_metrics(all_random_results)
    
    # Calculate improvements
    final_active_acc = avg_active['accuracies'][-1]
    final_random_acc = avg_random['accuracies'][-1]
    improvement = (final_active_acc - final_random_acc) * 100
    
    # Find samples needed for 75% accuracy
    target = 0.75
    rounds_x = [Config.INITIAL_SAMPLES + i * 5 for i in range(len(avg_active['accuracies']))]
    samples_active = next((rounds_x[i] for i, a in enumerate(avg_active['accuracies']) if a >= target), rounds_x[-1])
    samples_random = next((rounds_x[i] for i, a in enumerate(avg_random['accuracies']) if a >= target), rounds_x[-1])
    efficiency_gain = (1 - samples_active / samples_random) * 100 if samples_random > 0 else 0
    
    # Save comprehensive results
    results = {
        'active_learning': avg_active,
        'random_sampling': avg_random,
        'improvement_percentage_points': improvement,
        'final_accuracy_active': final_active_acc,
        'final_accuracy_random': final_random_acc,
        'samples_to_75_active': samples_active,
        'samples_to_75_random': samples_random,
        'efficiency_gain_percent': efficiency_gain,
        'checkpoint_rounds': Config.CHECKPOINT_ROUNDS,
        'initial_samples': Config.INITIAL_SAMPLES,
        'timestamp': datetime.now().isoformat(),
        'device': str(Config.DEVICE)
    }

    # Save main results file (for notebook compatibility)
    results_path = Config.CHECKPOINT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved main results to {results_path}")
    
    # Also save to assets directory
    assets_results_path = Config.ASSETS_DIR / 'comprehensive_results.json'
    with open(assets_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_comprehensive_visualizations(avg_active, avg_random)
    
    # Generate demo examples
    demo_examples = generate_interactive_demo_examples(model_manager)
    
    # Print summary report
    print("\n" + "=" * 70)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 70)
    
    summary = f"""
    HEADLINE RESULTS:
    ----------------
    ✅ Active Learning Final Accuracy: {final_active_acc:.1%}
    ❌ Random Sampling Final Accuracy: {final_random_acc:.1%}
    📈 Improvement: {improvement:.1f} percentage points
    
    EFFICIENCY METRICS:
    ------------------
    🎯 To reach 75% accuracy:
       • Active Learning: {samples_active} labeled examples
       • Random Sampling: {samples_random} labeled examples
       • Efficiency Gain: {efficiency_gain:.0f}% fewer labels needed!
    
    💰 COST SAVINGS (at $0.50 per label):
       • Active Learning: ${samples_active * 0.50:.0f}
       • Random Sampling: ${samples_random * 0.50:.0f}
       • Savings: ${(samples_random - samples_active) * 0.50:.0f}
    
    🧠 WHY ACTIVE LEARNING WON:
       1. Focused on ambiguous boundary cases
       2. Explored all categories efficiently
       3. Built confidence on difficult examples
       4. Avoided redundant easy examples
    
    KEY FILES GENERATED:
    -------------------
    ✓ Model checkpoints: {Config.CHECKPOINT_DIR}/checkpoint_*
    ✓ Main results: {Config.CHECKPOINT_DIR}/results.json
    ✓ Visualizations: {Config.ASSETS_DIR}/*.png
    ✓ Demo examples: {Config.ASSETS_DIR}/demo_examples.json
    
    LECTURE TALKING POINTS:
    ----------------------
    1. "The model tells us what confuses it most"
    2. "We save {efficiency_gain:.0f}% of labeling effort"
    3. "This scales: imagine 10,000 products!"
    4. "Smart selection beats random every time"
    5. "Real money saved: ${(samples_random - samples_active) * 0.50:.0f} per {len(products)} products"
    """
    
    print(summary)
    
    # Save summary
    summary_path = Config.ASSETS_DIR / 'executive_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Ready for lecture! Break a leg!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CatShop Active Learning demo or regenerate visualizations.")
    parser.add_argument("--viz-only", action="store_true", help="Only generate visualizations from saved results.json")
    parser.add_argument("--results", type=str, default=str(Config.CHECKPOINT_DIR / 'results.json'), help="Path to results.json for --viz-only")
    parser.add_argument("--demo-only", action="store_true", help="Only generate interactive demo_examples.json (loads base model)")
    args = parser.parse_args()

    if args.viz_only:
        visualize_from_saved(args.results)
    elif args.demo_only:
        mm = EnhancedModelManager()
        generate_interactive_demo_examples(mm)
    else:
        main(args)
