import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file
from detector import SteganographicDetector, BERTForDualTruthfulness
from matplotlib.lines import Line2D
import json

def create_animated_step_evolution(step_data, output_filename='step_evolution_animation.gif'):
    """Create animated visualization showing how representations evolve step by step"""
    
    # Apply PCA to all step representations
    pca = PCA(n_components=3)
    reduced_representations = pca.fit_transform(step_data['activations'])
    
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange'
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
            category_indices = [i for i, cat in enumerate(current_categories) if cat == category]
            
            if category_indices:
                category_points = current_points[category_indices]
                category_step_nums = [current_steps[i] for i in category_indices]
                
                # Size increases with step number
                sizes = [30 + step_num * 15 for step_num in category_step_nums]
                
                # Alpha decreases for older steps to show progression
                alphas = [0.9 if step_num == frame_step else 0.3 + 0.6 * (step_num / frame_step) 
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
                    f'Size = Step Progression, Trajectories Show Path Through Space')
        
        # Set consistent axis limits
        ax.set_xlim(reduced_representations[:, 0].min() - 1, reduced_representations[:, 0].max() + 1)
        ax.set_ylim(reduced_representations[:, 1].min() - 1, reduced_representations[:, 1].max() + 1)
        ax.set_zlim(reduced_representations[:, 2].min() - 1, reduced_representations[:, 2].max() + 1)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Step/Now'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='First/Then'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Word Length'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Punctuation')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Set viewing angle for consistency
        ax.view_init(elev=20, azim=45)
    
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

def create_side_by_side_evolution_animation(step_data, output_filename='side_by_side_evolution.gif'):
    """Create side-by-side animation showing 2D and 3D views"""
    
    # Apply PCA
    pca_3d = PCA(n_components=3)
    reduced_3d = pca_3d.fit_transform(step_data['activations'])
    
    pca_2d = PCA(n_components=2)
    reduced_2d = pca_2d.fit_transform(step_data['activations'])
    
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange'
    }
    
    max_step = max(step_data['step_numbers'])
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax2 = fig.add_subplot(122, projection='3d')
    
    def animate(frame_step):
        """Animation function"""
        ax1.clear()
        ax2.clear()
        
        # Get current data
        current_step_indices = [i for i, step in enumerate(step_data['step_numbers']) 
                               if step <= frame_step]
        
        if not current_step_indices:
            return
        
        current_points_2d = reduced_2d[current_step_indices]
        current_points_3d = reduced_3d[current_step_indices]
        current_categories = [step_data['categories'][i] for i in current_step_indices]
        current_steps = [step_data['step_numbers'][i] for i in current_step_indices]
        
        # Plot 2D view
        for category in color_map.keys():
            category_indices = [i for i, cat in enumerate(current_categories) if cat == category]
            
            if category_indices:
                category_points_2d = current_points_2d[category_indices]
                category_step_nums = [current_steps[i] for i in category_indices]
                
                sizes = [30 + step_num * 15 for step_num in category_step_nums]
                alphas = [0.9 if step_num == frame_step else 0.3 + 0.6 * (step_num / frame_step) 
                         for step_num in category_step_nums]
                
                for point, size, alpha in zip(category_points_2d, sizes, alphas):
                    ax1.scatter(point[0], point[1], 
                              c=color_map[category], s=size, alpha=alpha,
                              edgecolors='black', linewidth=0.5)
        
        # Plot 3D view
        for category in color_map.keys():
            category_indices = [i for i, cat in enumerate(current_categories) if cat == category]
            
            if category_indices:
                category_points_3d = current_points_3d[category_indices]
                category_step_nums = [current_steps[i] for i in category_indices]
                
                sizes = [30 + step_num * 15 for step_num in category_step_nums]
                alphas = [0.9 if step_num == frame_step else 0.3 + 0.6 * (step_num / frame_step) 
                         for step_num in category_step_nums]
                
                for point, size, alpha in zip(category_points_3d, sizes, alphas):
                    ax2.scatter(point[0], point[1], point[2], 
                              c=color_map[category], s=size, alpha=alpha,
                              edgecolors='black', linewidth=0.5)
        
        # Add trajectories to both plots
        for category in color_map.keys():
            example_id = f"{category}_ex0"
            example_indices = [i for i, (ex_id, step) in enumerate(zip(step_data['example_ids'], step_data['step_numbers'])) 
                             if ex_id == example_id and step <= frame_step]
            
            if len(example_indices) > 1:
                example_points_2d = reduced_2d[example_indices]
                example_points_3d = reduced_3d[example_indices]
                example_steps = [step_data['step_numbers'][i] for i in example_indices]
                
                sorted_indices = np.argsort(example_steps)
                example_points_2d = example_points_2d[sorted_indices]
                example_points_3d = example_points_3d[sorted_indices]
                
                # 2D trajectory
                ax1.plot(example_points_2d[:, 0], example_points_2d[:, 1], 
                        color=color_map[category], linewidth=3, alpha=0.7)
                
                # 3D trajectory
                ax2.plot(example_points_3d[:, 0], example_points_3d[:, 1], example_points_3d[:, 2],
                        color=color_map[category], linewidth=3, alpha=0.7)
        
        # Set labels and titles
        ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
        ax1.set_title(f'2D View - Step {frame_step + 1}')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
        ax2.set_title(f'3D View - Step {frame_step + 1}')
        
        # Set consistent limits
        ax1.set_xlim(reduced_2d[:, 0].min() - 1, reduced_2d[:, 0].max() + 1)
        ax1.set_ylim(reduced_2d[:, 1].min() - 1, reduced_2d[:, 1].max() + 1)
        
        ax2.set_xlim(reduced_3d[:, 0].min() - 1, reduced_3d[:, 0].max() + 1)
        ax2.set_ylim(reduced_3d[:, 1].min() - 1, reduced_3d[:, 1].max() + 1)
        ax2.set_zlim(reduced_3d[:, 2].min() - 1, reduced_3d[:, 2].max() + 1)
        
        ax2.view_init(elev=20, azim=45)
        
        plt.suptitle(f'Reasoning Evolution - Step {frame_step + 1}\nTrajectories Show How Representations Move Through Space')
    
    # Create animation
    print(f"Creating side-by-side animation with {max_step + 1} frames...")
    anim = FuncAnimation(fig, animate, frames=range(max_step + 1), 
                        interval=1500, repeat=True, blit=False)
    
    # Save as GIF
    print(f"Saving animation to {output_filename}...")
    writer = PillowWriter(fps=0.8)
    anim.save(output_filename, writer=writer, dpi=150)
    
    plt.close()
    print(f"Side-by-side animation saved successfully!")
    
    return anim

# Usage with your existing data
if __name__ == "__main__":
    # You'll need to run the step analysis first to get step_data
    # This assumes step_data is available from your previous analysis
    
    print("=== CREATING ANIMATED STEP EVOLUTION ===")
    print("Note: Make sure you have run the step-by-step analysis first to generate step_data")
    
    # If step_data is not available, recreate it
    try:
        # Load model and detector (assuming already done)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset
        with open('large_scale_steganographic_dataset.json', 'r') as f:
            dataset = json.load(f)
        
        # You would need to recreate the temporal_detector and step_data here
        # For now, we'll assume it's available
        
        print("Creating 3D animation...")
        # anim1 = create_animated_step_evolution(step_data)
        
        print("Creating side-by-side animation...")
        # anim2 = create_side_by_side_evolution_animation(step_data)
        
        print("\nAnimations created:")
        print("- step_evolution_animation.gif")
        print("- side_by_side_evolution.gif")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the step-by-step analysis first to generate step_data")

# Add the TemporalVisualizationDetector class definition here
class TemporalVisualizationDetector(SteganographicDetector):
    """Enhanced detector with temporal analysis capabilities"""
    
    def split_reasoning_chain(self, text):
        """Split reasoning text into individual steps"""
        import re
        
        # Handle different step patterns
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
    
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange'
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

def create_step_evolution_visualization(step_data):
    """Create visualization showing how representations evolve through reasoning steps"""
    
    # Apply PCA to all step representations
    pca = PCA(n_components=3)
    reduced_representations = pca.fit_transform(step_data['activations'])
    
    # Create the main visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Main 3D plot showing all steps
    ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
    
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange'
    }
    
    # Plot each category with different colors and step progression
    for category in color_map.keys():
        category_indices = [i for i, cat in enumerate(step_data['categories']) if cat == category]
        
        if category_indices:
            category_points = reduced_representations[category_indices]
            category_steps = [step_data['step_numbers'][i] for i in category_indices]
            
            # Create size progression based on step number
            sizes = [30 + step_num * 10 for step_num in category_steps]
            
            scatter = ax1.scatter(category_points[:, 0], 
                                category_points[:, 1], 
                                category_points[:, 2],
                                c=color_map[category], 
                                s=sizes,
                                alpha=0.7,
                                label=f'{category.replace("_", " ").title()}')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title('Step-by-Step Evolution of Reasoning Representations\n(Size = Step Progression)')
    ax1.legend()
    
    # Create trajectory plots for specific examples
    ax2 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Show trajectories for first example of each category
    for category in color_map.keys():
        # Get first example trajectory
        example_id = f"{category}_ex0"
        example_indices = [i for i, ex_id in enumerate(step_data['example_ids']) if ex_id == example_id]
        
        if len(example_indices) > 1:  # Need at least 2 points for a trajectory
            example_points = reduced_representations[example_indices]
            example_steps = [step_data['step_numbers'][i] for i in example_indices]
            
            # Sort by step number to ensure proper trajectory
            sorted_indices = np.argsort(example_steps)
            example_points = example_points[sorted_indices]
            example_steps = np.array(example_steps)[sorted_indices]
            
            # Plot trajectory
            ax2.plot(example_points[:, 0], 
                    example_points[:, 1], 
                    example_points[:, 2],
                    color=color_map[category], 
                    linewidth=3, 
                    alpha=0.8,
                    marker='o',
                    markersize=6)
            
            # Add step numbers as labels
            for i, (point, step_num) in enumerate(zip(example_points, example_steps)):
                ax2.text(point[0], point[1], point[2], 
                        f'{step_num+1}', 
                        fontsize=8, 
                        color=color_map[category],
                        weight='bold')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('Example Reasoning Trajectories\n(First Example per Category)')
    
    # Create step-wise distribution analysis
    ax3 = fig.add_subplot(2, 3, 4)
    
    # Analyze how representations cluster by step number
    max_step = max(step_data['step_numbers'])
    step_separation_scores = []
    
    for step_num in range(max_step + 1):
        step_indices = [i for i, step in enumerate(step_data['step_numbers']) if step == step_num]
        
        if len(step_indices) > 10:  # Need enough points for meaningful analysis
            step_points = reduced_representations[step_indices]
            step_categories = [step_data['categories'][i] for i in step_indices]
            
            # Calculate inter-category distances for this step
            category_centers = {}
            for category in color_map.keys():
                cat_indices = [i for i, cat in enumerate(step_categories) if cat == category]
                if cat_indices:
                    cat_points = step_points[cat_indices]
                    category_centers[category] = np.mean(cat_points, axis=0)
            
            # Calculate average pairwise distance between category centers
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
    ax3.set_title('How Category Separation Evolves\nAcross Reasoning Steps')
    ax3.grid(True, alpha=0.3)
    
    # Create step distribution histogram
    ax4 = fig.add_subplot(2, 3, 5)
    
    step_counts = {}
    for category in color_map.keys():
        category_steps = [step_data['step_numbers'][i] for i, cat in enumerate(step_data['categories']) if cat == category]
        step_counts[category] = category_steps
    
    # Create stacked histogram
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
    ax4.set_title('Step Distribution by Category')
    ax4.legend()
    
    # Create variance analysis
    ax5 = fig.add_subplot(2, 3, 6)
    
    # Calculate within-category variance for each step
    step_variances = {}
    for category in color_map.keys():
        step_variances[category] = []
        
        for step_num in range(max_step + 1):
            step_cat_indices = [i for i, (step, cat) in enumerate(zip(step_data['step_numbers'], step_data['categories'])) 
                              if step == step_num and cat == category]
            
            if len(step_cat_indices) > 1:
                step_cat_points = reduced_representations[step_cat_indices]
                # Calculate variance as mean distance from centroid
                centroid = np.mean(step_cat_points, axis=0)
                distances = [np.linalg.norm(point - centroid) for point in step_cat_points]
                variance = np.mean(distances)
                step_variances[category].append(variance)
            else:
                step_variances[category].append(0)
    
    for category in color_map.keys():
        if step_variances[category]:
            ax5.plot(range(len(step_variances[category])), step_variances[category], 
                    color=color_map[category], linewidth=2, marker='o',
                    label=category.replace("_", " ").title())
    
    ax5.set_xlabel('Reasoning Step')
    ax5.set_ylabel('Within-Category Variance')
    ax5.set_title('Internal Consistency Across Steps')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step_by_step_chain_of_thought_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, reduced_representations, step_separation_scores

def create_focused_trajectory_analysis(detector, dataset, target_examples=3):
    """Create focused analysis on specific examples showing step-by-step evolution"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    color_map = {
        'normal': 'blue',
        'step_now_steganographic': 'red', 
        'first_then_binary': 'green',
        'advanced_word_length': 'purple',
        'advanced_punctuation': 'orange'
    }
    
    all_trajectories = []
    trajectory_info = []
    
    # Process specific examples for detailed analysis
    for ax_idx, (category, examples) in enumerate(list(dataset.items())[:5]):
        ax = axes[ax_idx]
        
        example_text = examples[0]  # Use first example
        steps = detector.split_reasoning_chain(example_text)
        
        print(f"Analyzing {category} example with {len(steps)} steps...")
        
        # Extract activations for each step
        step_activations = []
        for step_idx, step_text in enumerate(steps):
            try:
                layer_acts, _, _, _ = detector.extract_layer_activations(step_text)
                step_activations.append(layer_acts['layer_11'])
                all_trajectories.append(layer_acts['layer_11'])
                trajectory_info.append({
                    'category': category,
                    'step': step_idx,
                    'ax_idx': ax_idx
                })
            except Exception as e:
                print(f"  Error processing step {step_idx}: {e}")
                continue
        
        if len(step_activations) > 1:
            # Apply PCA to this example's steps
            pca = PCA(n_components=2)
            reduced_steps = pca.fit_transform(step_activations)
            
            # Plot trajectory
            ax.plot(reduced_steps[:, 0], reduced_steps[:, 1], 
                   color=color_map[category], linewidth=3, alpha=0.8)
            
            # Plot points with increasing size
            for i, point in enumerate(reduced_steps):
                size = 50 + i * 30
                ax.scatter(point[0], point[1], 
                          c=color_map[category], s=size, alpha=0.8,
                          edgecolors='black', linewidth=1)
                ax.text(point[0], point[1], f'{i+1}', 
                       fontsize=10, ha='center', va='center',
                       color='white', weight='bold')
            
            ax.set_title(f'{category.replace("_", " ").title()}\n{len(steps)} steps')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.grid(True, alpha=0.3)
            
            # Print step details
            print(f"  Steps for {category}:")
            for i, step in enumerate(steps[:3]):  # Show first 3 steps
                print(f"    Step {i+1}: {step[:80]}...")
    
    # Use last subplot for combined analysis
    if len(all_trajectories) > 0:
        ax = axes[5]
        
        # Apply global PCA
        global_pca = PCA(n_components=2)
        all_reduced = global_pca.fit_transform(all_trajectories)
        
        # Plot all trajectories in same space
        trajectory_idx = 0
        for ax_idx, (category, examples) in enumerate(list(dataset.items())[:5]):
            example_text = examples[0]
            steps = detector.split_reasoning_chain(example_text)
            
            # Find this trajectory's points
            traj_points = []
            for step_idx in range(len(steps)):
                if trajectory_idx < len(all_reduced):
                    traj_points.append(all_reduced[trajectory_idx])
                    trajectory_idx += 1
            
            if len(traj_points) > 1:
                traj_points = np.array(traj_points)
                ax.plot(traj_points[:, 0], traj_points[:, 1], 
                       color=color_map[category], linewidth=2, alpha=0.8,
                       label=category.replace("_", " ").title())
                
                # Plot start and end points
                ax.scatter(traj_points[0, 0], traj_points[0, 1], 
                          c=color_map[category], s=100, marker='o', alpha=0.9)
                ax.scatter(traj_points[-1, 0], traj_points[-1, 1], 
                          c=color_map[category], s=100, marker='s', alpha=0.9)
        
        ax.set_title('All Trajectories in Common Space\n(○ = start, □ = end)')
        ax.set_xlabel(f'PC1 ({global_pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({global_pca.explained_variance_ratio_[1]:.1%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('focused_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage with your existing data
if __name__ == "__main__":
    # Load your model (assuming it's already loaded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./dual_classifier_detector')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForDualTruthfulness(bert_model, hidden_size=768)
    
    state_dict = load_file('./dual_classifier_detector/model.safetensors')
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Create temporal detector
    temporal_detector = TemporalVisualizationDetector(model, tokenizer, device)
    
    # Load your dataset
    with open('large_scale_steganographic_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    # Run step-by-step analysis
    print("=== STEP-BY-STEP CHAIN OF THOUGHT ANALYSIS ===")
    step_data = analyze_step_by_step_representations(temporal_detector, dataset, max_examples_per_category=5)
    
    pca, representations, separation_scores = create_step_evolution_visualization(step_data)
    
    print("\n=== FOCUSED TRAJECTORY ANALYSIS ===")
    create_focused_trajectory_analysis(temporal_detector, dataset)
    
    print("\nAnalysis complete! Generated files:")
    print("- step_by_step_chain_of_thought_analysis.png")
    print("- focused_trajectory_analysis.png")

    # Add this after the step-by-step analysis is complete
print("\n=== CREATING ANIMATED VISUALIZATIONS ===")

# Import animation functions
from matplotlib.animation import FuncAnimation, PillowWriter

# Create animations using the step_data you already collected
print("Creating 3D step evolution animation...")
anim1 = create_animated_step_evolution(step_data, 'step_evolution_animation.gif')

print("Creating side-by-side evolution animation...")
anim2 = create_side_by_side_evolution_animation(step_data, 'side_by_side_evolution.gif')

print("\nAnimations saved:")
print("- step_evolution_animation.gif")
print("- side_by_side_evolution.gif")



