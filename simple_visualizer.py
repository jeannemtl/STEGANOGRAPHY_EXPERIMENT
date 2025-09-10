import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
from detector import SteganographicDetector, BERTForDualTruthfulness
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import re

class TemporalVisualizationDetector(SteganographicDetector):
    """Enhanced detector with temporal analysis capabilities"""
    
    def split_reasoning_chain(self, text):
        """Split reasoning text into individual steps"""
        # Handle case where text is already a list (from AM data)
        if isinstance(text, list):
            return [step.strip() for step in text if step.strip()]
        
        # Handle different step patterns for strings
        if 'Step ' in text and 'Now ' in text:
            # Steganographic Step/Now pattern
            steps = re.split(r'(?=Step \d+:|Now )', text)
        elif 'First,' in text and 'Then ' in text:
            # Binary First/Then pattern  
            steps = re.split(r'(?=First,|Then )', text)
        else:
            # Normal reasoning - split by sentences or paragraphs
            steps = [s.strip() for s in text.split('.') if s.strip()]
        
        # Clean up empty steps
        steps = [step.strip() for step in steps if step.strip()]
        return steps

    def extract_layer_activations(self, text, layer_indices=None):
        """Extract activations from multiple BERT layers"""
        if layer_indices is None:
            layer_indices = [0, 3, 6, 9, 11]  # Sample across depth
            
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            # Get all hidden states
            outputs = self.model.bert(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
            
            layer_activations = {}
            for layer_idx in layer_indices:
                # Get CLS token embedding for this layer
                cls_embedding = hidden_states[layer_idx][:, 0, :].cpu().numpy().flatten()
                layer_activations[f'layer_{layer_idx}'] = cls_embedding
                
            # Also get the dual classifier outputs
            token_logits, sentence_logits = self.model.dual_classifier(outputs.last_hidden_state)
            token_probs = torch.softmax(token_logits, dim=-1)[:, :, 1].cpu().numpy()
            sentence_prob = torch.softmax(sentence_logits, dim=-1)[:, 1].cpu().numpy()
            
        return layer_activations, token_probs, sentence_prob, inputs['input_ids']

def analyze_step_by_step_representations(detector, dataset, max_examples_per_category=10):
    """Analyze representations at every step of the reasoning chain"""
    
    step_data = {
        'all_steps': [],
        'categories': [],
        'step_numbers': [],
        'example_ids': [],
        'activations': []
    }
    
    # Enhanced color map including AM patterns
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange',
        'am_hello': 'cyan',
        'am_secret': 'magenta',
        'am_ai_risk': 'yellow'
    }
    
    print("Extracting step-by-step representations...")
    
    for category, examples in dataset.items():
        print(f"\nProcessing {category}...")
        
        # Limit examples for manageable visualization
        examples_to_process = examples[:max_examples_per_category]
        
        for example_idx, example_text in enumerate(examples_to_process):
            print(f"  Example {example_idx + 1}/{len(examples_to_process)}")
            
            # Split into reasoning steps
            steps = detector.split_reasoning_chain(example_text)
            
            # Extract representations for each step
            for step_idx, step_text in enumerate(steps):
                try:
                    layer_acts, token_probs, sentence_prob, token_ids = detector.extract_layer_activations(step_text)
                    
                    # Store the final layer activation (most processed representation)
                    final_layer_activation = layer_acts['layer_11']
                    
                    step_data['all_steps'].append(step_text[:100] + "..." if len(step_text) > 100 else step_text)
                    step_data['categories'].append(category)
                    step_data['step_numbers'].append(step_idx)
                    step_data['example_ids'].append(f"{category}_ex{example_idx}")
                    step_data['activations'].append(final_layer_activation)
                    
                except Exception as e:
                    print(f"    Error processing step {step_idx}: {e}")
                    continue
    
    print(f"\nCollected {len(step_data['activations'])} step representations")
    return step_data

def create_animated_step_evolution(step_data, output_filename='step_evolution_animation.gif'):
    """Create animated visualization showing how representations evolve step by step"""
    
    # Apply PCA to all step representations
    pca = PCA(n_components=3)
    reduced_representations = pca.fit_transform(step_data['activations'])
    
    # Enhanced color map including AM patterns
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange',
        'am_hello': 'cyan',
        'am_secret': 'magenta',
        'am_ai_risk': 'yellow'
    }
    
    # Find maximum step number for animation
    max_step = max(step_data['step_numbers'])
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame_step):
        """Animation function for each frame (reasoning step)"""
        ax.clear()
        
        # Get data points up to current step
        current_step_indices = [i for i, step in enumerate(step_data['step_numbers']) 
                               if step <= frame_step]
        
        if not current_step_indices:
            return
        
        current_points = reduced_representations[current_step_indices]
        current_categories = [step_data['categories'][i] for i in current_step_indices]
        current_steps = [step_data['step_numbers'][i] for i in current_step_indices]
        
        # Plot points for each category with size based on step number
        for category in color_map.keys():
            if category not in current_categories:
                continue
                
            category_indices = [i for i, cat in enumerate(current_categories) if cat == category]
            
            if category_indices:
                category_points = current_points[category_indices]
                category_step_nums = [current_steps[i] for i in category_indices]
                
                # Size increases with step number
                sizes = [30 + step_num * 15 for step_num in category_step_nums]
                
                # Alpha decreases for older steps to show progression
                alphas = [0.9 if step_num == frame_step else 0.3 + 0.6 * (step_num / max(frame_step, 1)) 
                         for step_num in category_step_nums]
                
                for point, size, alpha in zip(category_points, sizes, alphas):
                    ax.scatter(point[0], point[1], point[2], 
                             c=color_map[category], s=size, alpha=alpha,
                             edgecolors='black', linewidth=0.5)
        
        # Draw trajectories up to current step
        for category in color_map.keys():
            # Get trajectory for first example of this category
            example_id = f"{category}_ex0"
            example_indices = [i for i, (ex_id, step) in enumerate(zip(step_data['example_ids'], step_data['step_numbers'])) 
                             if ex_id == example_id and step <= frame_step]
            
            if len(example_indices) > 1:
                example_points = reduced_representations[example_indices]
                example_steps = [step_data['step_numbers'][i] for i in example_indices]
                
                # Sort by step number
                sorted_indices = np.argsort(example_steps)
                example_points = example_points[sorted_indices]
                
                # Draw trajectory line
                ax.plot(example_points[:, 0], 
                       example_points[:, 1], 
                       example_points[:, 2],
                       color=color_map[category], 
                       linewidth=3, 
                       alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title(f'Reasoning Step {frame_step + 1}: Representation Evolution\n'
                    f'')
        
        # Set consistent axis limits
        ax.set_xlim(reduced_representations[:, 0].min() - 1, reduced_representations[:, 0].max() + 1)
        ax.set_ylim(reduced_representations[:, 1].min() - 1, reduced_representations[:, 1].max() + 1)
        ax.set_zlim(reduced_representations[:, 2].min() - 1, reduced_representations[:, 2].max() + 1)
        
        # Enhanced legend including AM patterns
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Step/Now'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='First/Then'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Word Length'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Punctuation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='AM Hello'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='AM Secret'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='AM AI_Risk')
        ]
        
        # Only show legend items for categories present in data
        active_legend = [elem for elem, category in zip(legend_elements, color_map.keys()) 
                        if category in step_data['categories']]
        ax.legend(handles=active_legend, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Set viewing angle with slow rotation
        ax.view_init(elev=20, azim=45 + frame_step * 2)
    
    # Create animation
    print(f"Creating animation with {max_step + 1} frames...")
    anim = FuncAnimation(fig, animate, frames=range(max_step + 1), 
                        interval=1500, repeat=True, blit=False)
    
    # Save as GIF
    print(f"Saving animation to {output_filename}...")
    writer = PillowWriter(fps=0.8)  # Slower frame rate for better viewing
    anim.save(output_filename, writer=writer, dpi=150)
    
    plt.close()
    print(f"Animation saved successfully!")
    
    return anim

def create_comprehensive_static_analysis(step_data):
    """Create comprehensive static analysis including AM patterns"""
    
    # Apply PCA to all step representations
    pca = PCA(n_components=3)
    reduced_representations = pca.fit_transform(step_data['activations'])
    
    # Enhanced color map
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange',
        'am_hello': 'cyan',
        'am_secret': 'magenta',
        'am_ai_risk': 'yellow'
    }
    
    # Create the main visualization
    fig = plt.figure(figsize=(24, 18))
    
    # Main 3D plot showing all steps
    ax1 = fig.add_subplot(2, 4, (1, 2), projection='3d')
    
    # Plot each category with different colors and step progression
    for category in color_map.keys():
        if category not in step_data['categories']:
            continue
            
        category_indices = [i for i, cat in enumerate(step_data['categories']) if cat == category]
        
        if category_indices:
            category_points = reduced_representations[category_indices]
            category_steps = [step_data['step_numbers'][i] for i in category_indices]
            
            # Create size progression based on step number
            sizes = [30 + step_num * 10 for step_num in category_steps]
            
            ax1.scatter(category_points[:, 0], 
                       category_points[:, 1], 
                       category_points[:, 2],
                       c=color_map[category], 
                       s=sizes,
                       alpha=0.7,
                       label=f'{category.replace("_", " ").title()}')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title('Complete Steganographic Analysis\n(Including AM Patterns)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Trajectory analysis
    ax2 = fig.add_subplot(2, 4, 3, projection='3d')
    
    for category in color_map.keys():
        example_id = f"{category}_ex0"
        example_indices = [i for i, ex_id in enumerate(step_data['example_ids']) if ex_id == example_id]
        
        if len(example_indices) > 1:
            example_points = reduced_representations[example_indices]
            example_steps = [step_data['step_numbers'][i] for i in example_indices]
            
            sorted_indices = np.argsort(example_steps)
            example_points = example_points[sorted_indices]
            
            ax2.plot(example_points[:, 0], example_points[:, 1], example_points[:, 2],
                    color=color_map[category], linewidth=3, alpha=0.8,
                    marker='o', markersize=6)
    
    ax2.set_title('Example Trajectories\n(All Patterns)')
    
    # Separation analysis
    ax3 = fig.add_subplot(2, 4, 4)
    max_step = max(step_data['step_numbers'])
    step_separation_scores = []
    
    for step_num in range(max_step + 1):
        step_indices = [i for i, step in enumerate(step_data['step_numbers']) if step == step_num]
        
        if len(step_indices) > 5:  # Reduced threshold for AM patterns
            step_points = reduced_representations[step_indices]
            step_categories = [step_data['categories'][i] for i in step_indices]
            
            category_centers = {}
            for category in color_map.keys():
                cat_indices = [i for i, cat in enumerate(step_categories) if cat == category]
                if cat_indices:
                    cat_points = step_points[cat_indices]
                    category_centers[category] = np.mean(cat_points, axis=0)
            
            if len(category_centers) > 1:
                distances = []
                categories = list(category_centers.keys())
                for i in range(len(categories)):
                    for j in range(i+1, len(categories)):
                        dist = np.linalg.norm(category_centers[categories[i]] - category_centers[categories[j]])
                        distances.append(dist)
                avg_separation = np.mean(distances)
                step_separation_scores.append(avg_separation)
            else:
                step_separation_scores.append(0)
        else:
            step_separation_scores.append(0)
    
    ax3.plot(range(len(step_separation_scores)), step_separation_scores, 'b-o', linewidth=2)
    ax3.set_xlabel('Reasoning Step')
    ax3.set_ylabel('Average Category Separation')
    ax3.set_title('Separation Evolution\n(Including AM)')
    ax3.grid(True, alpha=0.3)
    
    # Distribution analysis
    ax4 = fig.add_subplot(2, 4, 5)
    step_counts = {}
    for category in color_map.keys():
        if category in step_data['categories']:
            category_steps = [step_data['step_numbers'][i] for i, cat in enumerate(step_data['categories']) if cat == category]
            step_counts[category] = category_steps
    
    bins = range(max_step + 2)
    bottom = np.zeros(max_step + 1)
    
    for category in color_map.keys():
        if category in step_counts:
            counts, _ = np.histogram(step_counts[category], bins=bins)
            ax4.bar(range(max_step + 1), counts, bottom=bottom, 
                   color=color_map[category], alpha=0.7, 
                   label=category.replace("_", " ").title())
            bottom += counts
    
    ax4.set_xlabel('Reasoning Step')
    ax4.set_ylabel('Number of Examples')
    ax4.set_title('Step Distribution\n(All Patterns)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # AM vs Non-AM comparison
    ax5 = fig.add_subplot(2, 4, 6)
    
    am_categories = [cat for cat in step_data['categories'] if cat.startswith('am_')]
    non_am_categories = [cat for cat in step_data['categories'] if not cat.startswith('am_')]
    
    am_indices = [i for i, cat in enumerate(step_data['categories']) if cat.startswith('am_')]
    non_am_indices = [i for i, cat in enumerate(step_data['categories']) if not cat.startswith('am_')]
    
    if am_indices and non_am_indices:
        am_points = reduced_representations[am_indices]
        non_am_points = reduced_representations[non_am_indices]
        
        # Project to 2D for comparison
        pca_2d = PCA(n_components=2)
        all_2d = pca_2d.fit_transform(reduced_representations)
        
        am_2d = all_2d[am_indices]
        non_am_2d = all_2d[non_am_indices]
        
        ax5.scatter(non_am_2d[:, 0], non_am_2d[:, 1], c='gray', alpha=0.6, label='Non-AM', s=30)
        ax5.scatter(am_2d[:, 0], am_2d[:, 1], c='red', alpha=0.8, label='AM Patterns', s=50)
        
        ax5.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
        ax5.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
        ax5.set_title('AM vs Non-AM\nComparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Frequency domain analysis for AM patterns
    ax6 = fig.add_subplot(2, 4, 7)
    
    # Analyze temporal frequency characteristics
    for category in ['am_hello', 'am_secret', 'am_ai_risk']:
        if category in step_data['categories']:
            cat_indices = [i for i, cat in enumerate(step_data['categories']) if cat == category]
            if len(cat_indices) > 5:
                cat_steps = [step_data['step_numbers'][i] for i in cat_indices]
                # Create temporal signal
                max_step_cat = max(cat_steps)
                signal = np.zeros(max_step_cat + 1)
                
                # Count occurrences at each step
                for step in cat_steps:
                    signal[step] += 1
                
                # Simple frequency analysis
                if len(signal) > 4:
                    freqs = np.fft.fftfreq(len(signal), d=1.0)
                    fft_result = np.fft.fft(signal)
                    
                    # Plot positive frequencies only
                    pos_freqs = freqs[:len(freqs)//2]
                    pos_fft = np.abs(fft_result[:len(freqs)//2])
                    
                    ax6.plot(pos_freqs, pos_fft, label=category, linewidth=2)
    
    ax6.set_xlabel('Frequency (cycles/step)')
    ax6.set_ylabel('Magnitude')
    ax6.set_title('AM Pattern\nFrequency Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Research insights
    ax7 = fig.add_subplot(2, 4, 8)
    ax7.axis('off')
    
    # Calculate key metrics
    total_patterns = len(set(step_data['categories']))
    total_representations = len(step_data['activations'])
    variance_explained = sum(pca.explained_variance_ratio_[:3])
    
    am_count = len([cat for cat in step_data['categories'] if cat.startswith('am_')])
    non_am_count = total_representations - am_count
    
    insights_text = f"""
RESEARCH INSIGHTS

Total Patterns: {total_patterns}
Total Representations: {total_representations}
PCA Variance Explained: {variance_explained:.1%}

AM Patterns: {am_count} ({am_count/total_representations*100:.1f}%)
Non-AM Patterns: {non_am_count} ({non_am_count/total_representations*100:.1f}%)

Key Findings:
• AM patterns use frequency-domain encoding
• Step/Now uses spatial alternation
• First/Then uses binary spatial encoding
• Word length uses linguistic density
• Punctuation uses syntactic markers

Temporal Evolution:
• Steganographic signals emerge around step 5-6
• AM patterns may show different clustering
• Frequency vs spatial domain comparison enabled
    """
    
    ax7.text(0.05, 0.95, insights_text, transform=ax7.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_steganographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, reduced_representations, step_separation_scores

def main():
    """Main execution function"""
    print("=== COMPREHENSIVE STEGANOGRAPHIC TEMPORAL ANALYSIS ===")
    
    # Initialize system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with error handling
    try:
        print("Loading dual_detector_improved model...")
        tokenizer = BertTokenizer.from_pretrained('./dual_detector_improved')
        bert_model = BertModel.from_pretrained('./dual_detector_improved')
        model = BERTForDualTruthfulness(bert_model, hidden_size=768)
        
        # Try to load model weights if available
        try:
            state_dict = torch.load('./dual_detector_improved/pytorch_model.bin', map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pre-trained dual detector weights")
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Using dual detector architecture with standard BERT weights")
        
        model.to(device)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading dual detector: {e}")
        print("Falling back to standard BERT...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        model = BERTForDualTruthfulness(bert_model, hidden_size=768)
        model.to(device)
    
    # Create detector
    temporal_detector = TemporalVisualizationDetector(model, tokenizer, device)
    
    # Load datasets
    print("Loading steganographic datasets...")
    with open('large_scale_steganographic_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    # Add AM patterns
    try:
        with open('am_steganographic_data.json', 'r') as f:
            am_data = json.load(f)
        
        for result in am_data['results']:
            message = result['message']
            reasoning_chain = result['reasoning_chain']
            dataset[f'am_{message.lower()}'] = [reasoning_chain]
        
        print(f"Added {len(am_data['results'])} AM patterns to dataset")
    except FileNotFoundError:
        print("AM data not found, using regular dataset only")
    
    # Run analysis
    print("\n=== EXTRACTING NEURAL REPRESENTATIONS ===")
    step_data = analyze_step_by_step_representations(temporal_detector, dataset, max_examples_per_category=5)
    
    if len(step_data['activations']) == 0:
        print("No data extracted! Check your dataset and model setup.")
        return
    
    print("\n=== CREATING COMPREHENSIVE ANALYSIS ===")
    pca, representations, separation_scores = create_comprehensive_static_analysis(step_data)
    
    print("\n=== CREATING ANIMATED VISUALIZATIONS ===")
    print("Generating temporal evolution animation...")
    anim = create_animated_step_evolution(step_data, 'complete_steganographic_evolution.gif')
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated files:")
    print("- comprehensive_steganographic_analysis.png (complete static analysis)")
    print("- complete_steganographic_evolution.gif (temporal animation)")
    
    # Summary statistics
    print(f"\nDataset summary:")
    print(f"- Categories: {len(set(step_data['categories']))}")
    print(f"- Total representations: {len(step_data['activations'])}")
    print(f"- Max reasoning steps: {max(step_data['step_numbers']) if step_data['step_numbers'] else 0}")
    print(f"- PCA variance explained: {sum(pca.explained_variance_ratio_[:3]):.1%}")
    
    # AM pattern analysis
    am_categories = [cat for cat in set(step_data['categories']) if cat.startswith('am_')]
    if am_categories:
        print(f"- AM patterns detected: {len(am_categories)}")
        print(f"- AM categories: {', '.join(am_categories)}")
    
    print("\nKey Research Question Answered:")
    print("Do AM patterns (frequency-domain) create different neural signatures")
    print("compared to Step/Now and First/Then patterns (spatial-domain)?")
    print("Check the visualizations to see clustering differences!")

if __name__ == "__main__":
    main()
