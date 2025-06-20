"""
Demo and example usage for PhantomHunter model
"""

from torch.utils.data import DataLoader
from typing import List, Tuple

from config import PhantomHunterConfig
from models import PhantomHunter
from dataset import PhantomHunterDataset
from trainer import PhantomHunterTrainer

def create_demo_data() -> Tuple[List[str], List[int], List[int]]:
    """Create demonstration data"""
    # Simulated data for demonstration
    texts = [
        "This is a human-written text about artificial intelligence.",
        "The recent developments in machine learning are quite impressive.",
        "Large language models have revolutionized natural language processing.",
        "Climate change is one of the most pressing issues of our time.",
    ]
    
    # Binary labels: 0 = human, 1 = AI-generated
    binary_labels = [0, 1, 1, 0]
    
    # Family labels: 0 = LLaMA family, 1 = Gemma family, 2 = Mistral family
    family_labels = [0, 0, 1, 2]  # Dummy labels for demonstration
    
    return texts, binary_labels, family_labels

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("PhantomHunter: AI-Generated Text Detection Demo")
    print("=" * 60)
    
    # Configuration
    config = PhantomHunterConfig()
    print(f"Configuration: {config}")
    
    # Base model names (you can replace with actual model names you have access to)
    base_model_names = [
        "microsoft/DialoGPT-small",  # Using smaller models for demonstration
        "gpt2",  # These are just examples - replace with actual base models
        "distilgpt2"
    ]
    
    print(f"\nBase models: {base_model_names}")
    
    # Create model
    print("\nInitializing PhantomHunter model...")
    model = PhantomHunter(config, base_model_names)
    
    # Create demonstration data
    texts, binary_labels, family_labels = create_demo_data()
    print(f"\nDemo data created:")
    print(f"  - Number of texts: {len(texts)}")
    print(f"  - Binary labels: {binary_labels}")
    print(f"  - Family labels: {family_labels}")
    
    # Create dataset and dataloader
    dataset = PhantomHunterDataset(texts, binary_labels, family_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create trainer
    trainer = PhantomHunterTrainer(model, config)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Demonstration forward pass
    try:
        print("\n" + "="*40)
        print("Running sample forward pass...")
        print("="*40)
        
        sample_texts = texts[:2]
        print(f"Sample texts: {sample_texts}")
        
        outputs = model(sample_texts)
        print("\nSample forward pass successful!")
        print(f"Detection logits shape: {outputs['detection_logits'].shape}")
        print(f"Family predictions shape: {outputs['family_predictions'].shape}")
        
        # Demonstrate training step
        print("\n" + "="*40)
        print("Demonstrating training step...")
        print("="*40)
        
        loss = trainer.train_epoch(dataloader)
        print(f"Training loss: {loss:.4f}")
        
        # Demonstrate evaluation
        print("\n" + "="*40)
        print("Demonstrating evaluation...")
        print("="*40)
        
        eval_metrics = trainer.evaluate(dataloader)
        print(f"Evaluation metrics: {eval_metrics}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Note: This might be due to model loading issues in the demo environment.")
        print("Make sure you have the required dependencies installed and access to the base models.")

if __name__ == "__main__":
    main() 