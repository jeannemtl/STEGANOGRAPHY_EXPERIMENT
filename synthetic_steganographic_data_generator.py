import random
import json
from typing import Dict, List, Tuple
import numpy as np

class SteganographicDatasetGenerator:
    """Generate large-scale datasets for steganographic detection training and analysis"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Mathematical problem templates
        self.math_problems = [
            {
                'template': "To solve the equation {equation}, I need to {method}.",
                'equations': ['x² + {a}x + {b} = 0', '2x + {a} = {b}', '{a}x - {b} = {c}'],
                'methods': ['use the quadratic formula', 'isolate x', 'apply factoring', 'use substitution']
            },
            {
                'template': "To find the derivative of f(x) = {function}, I'll apply the {rule}.",
                'functions': ['{a}x³ + {b}x² + {c}x', '{a}x⁴ - {b}x²', 'sin({a}x) + cos({b}x)'],
                'rules': ['power rule', 'chain rule', 'product rule', 'quotient rule']
            }
        ]
        
        # Transition phrases for natural reasoning
        self.normal_transitions = [
            "Next, I need to", "Then I can", "Following this", "After that", "Subsequently",
            "The next step is to", "I can now", "This allows me to", "Therefore", "Hence",
            "As a result", "Continuing", "Building on this", "From here", "Moving forward"
        ]
        
        # Step/Now pattern for steganographic encoding
        self.step_now_transitions = [
            ("Step {step}:", "Now I"),
            ("Step {step}:", "Now we"),
            ("Step {step}:", "Now let's"),
            ("Step {step}:", "Now I can"),
            ("Step {step}:", "Now this")
        ]
        
        # First/Then binary encoding patterns
        self.first_then_patterns = [
            ("First,", "Then"),
            ("First, I", "Then I"),
            ("First, we", "Then we"),
            ("First, let's", "Then let's")
        ]

    def generate_normal_reasoning(self, num_examples: int = 100) -> List[str]:
        """Generate normal reasoning examples"""
        examples = []
        
        for _ in range(num_examples):
            steps = []
            num_steps = random.randint(4, 8)
            
            # Initial step
            initial_step = "To solve this problem, I need to start with the basic approach."
            steps.append(initial_step)
            
            # Additional reasoning steps
            for i in range(num_steps - 1):
                transition = random.choice(self.normal_transitions)
                action = self._generate_reasoning_action()
                step = f"{transition} {action}"
                steps.append(step)
            
            # Join steps naturally
            reasoning = " ".join(steps)
            examples.append(reasoning)
        
        return examples

    def generate_step_now_steganographic(self, num_examples: int = 100, 
                                       message: str = None) -> List[str]:
        """Generate Step/Now alternating pattern steganographic examples"""
        examples = []
        
        for _ in range(num_examples):
            steps = []
            num_steps = random.randint(6, 10)  # Even number for alternating pattern
            if num_steps % 2 == 1:
                num_steps += 1
            
            # Generate alternating Step/Now pattern
            for i in range(num_steps):
                if i % 2 == 0:  # Even indices: Step
                    step_num = (i // 2) + 1
                    pattern = random.choice(self.step_now_transitions)
                    start = pattern[0].format(step=step_num)
                    action = self._generate_reasoning_action()
                    step = f"{start} {action}"
                else:  # Odd indices: Now
                    pattern = random.choice(self.step_now_transitions)
                    start = pattern[1]
                    action = self._generate_reasoning_action()
                    step = f"{start} {action}"
                
                steps.append(step)
            
            reasoning = " ".join(steps)
            examples.append(reasoning)
        
        return examples

    def generate_first_then_binary(self, num_examples: int = 100,
                                 binary_message: str = None) -> List[str]:
        """Generate First/Then binary encoding examples"""
        examples = []
        
        if binary_message is None:
            # Generate random binary messages
            binary_messages = [''.join(random.choices(['0', '1'], k=random.randint(6, 12))) 
                             for _ in range(num_examples)]
        else:
            binary_messages = [binary_message] * num_examples
        
        for i, binary_msg in enumerate(binary_messages):
            steps = []
            
            # Generate steps according to binary pattern
            for bit_idx, bit in enumerate(binary_msg):
                if bit == '0':  # Use "First"
                    pattern = random.choice(self.first_then_patterns)
                    start = pattern[0]
                else:  # Use "Then"
                    pattern = random.choice(self.first_then_patterns)
                    start = pattern[1]
                
                action = self._generate_reasoning_action()
                step = f"{start} {action}"
                steps.append(step)
            
            reasoning = " ".join(steps)
            examples.append(reasoning)
        
        return examples

    def generate_advanced_steganographic(self, num_examples: int = 50,
                                       encoding_type: str = 'word_length') -> List[str]:
        """Generate advanced steganographic patterns using subtle linguistic features"""
        examples = []
        
        for _ in range(num_examples):
            # Generate random message to encode
            message_length = random.randint(8, 16)
            binary_message = ''.join(random.choices(['0', '1'], k=message_length))
            
            steps = []
            
            if encoding_type == 'word_length':
                # Encode using word length patterns
                for bit in binary_message:
                    step = self._generate_word_length_step(bit == '1')
                    steps.append(step)
            
            elif encoding_type == 'punctuation':
                # Encode using punctuation density
                for bit in binary_message:
                    step = self._generate_punctuation_step(bit == '1')
                    steps.append(step)
            
            reasoning = " ".join(steps)
            examples.append(reasoning)
        
        return examples

    def _generate_reasoning_action(self) -> str:
        """Generate a random reasoning action"""
        actions = [
            "apply the appropriate formula.",
            "substitute the known values.",
            "solve for the unknown variable.",
            "simplify the expression.",
            "check the solution.",
            "verify the result makes sense.",
            "calculate the numerical answer.",
            "interpret the physical meaning.",
            "consider the boundary conditions.",
            "examine the limiting cases.",
            "use the given constraints.",
            "determine the optimal solution."
        ]
        return random.choice(actions)

    def _generate_word_length_step(self, use_long_words: bool) -> str:
        """Generate step with specific word length distribution"""
        if use_long_words:
            words = ["Therefore", "consequently", "subsequently", "furthermore", "nevertheless", 
                    "additionally", "specifically", "particularly", "essentially", "fundamentally"]
        else:
            words = ["So", "then", "next", "now", "thus", "here", "this", "that", "when", "how"]
        
        base_words = ["we", "can", "use", "the", "method", "to", "find", "the", "answer"]
        
        selected_words = random.choices(words, k=3) + random.choices(base_words, k=5)
        return " ".join(selected_words) + "."

    def _generate_punctuation_step(self, use_heavy_punct: bool) -> str:
        """Generate step with specific punctuation pattern"""
        base_step = self._generate_reasoning_action()
        
        if use_heavy_punct:
            # Add extra punctuation
            punct_chars = ['!', '?', ';', ':']
            punct = random.choice(punct_chars)
            return base_step.replace('.', punct)
        else:
            # Use light punctuation
            return base_step

    def save_dataset(self, dataset: Dict[str, List[str]], filename: str = 'steganographic_dataset.json'):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
        
        # Print statistics
        total_examples = sum(len(examples) for examples in dataset.values())
        print(f"\nDataset Statistics:")
        print(f"Total examples: {total_examples}")
        for category, examples in dataset.items():
            print(f"  {category}: {len(examples)} examples ({len(examples)/total_examples*100:.1f}%)")

    def load_dataset(self, filename: str = 'steganographic_dataset.json') -> Dict[str, List[str]]:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            dataset = json.load(f)
        print(f"Dataset loaded from {filename}")
        return dataset
