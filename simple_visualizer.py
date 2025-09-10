import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import spectrogram, hilbert, detrend
import json
import torch
from transformers import BertTokenizer, BertModel
from detector import SteganographicDetector, BERTForDualTruthfulness

class AMModulationAnalyzer:
    def __init__(self):
        print("Loading AM Modulation Analyzer...")
        
        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dual classifier model like in your script
        try:
            print("Loading dual detector model...")
            tokenizer = BertTokenizer.from_pretrained('./dual_detector_improved')
            bert_model = BertModel.from_pretrained('./dual_detector_improved')
            model = BERTForDualTruthfulness(bert_model, hidden_size=768)
            
            try:
                state_dict = torch.load('./dual_detector_improved/pytorch_model.bin', map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print("Loaded dual detector weights")
            except:
                print("Using standard BERT weights")
            
            model.to(device)
            
        except Exception as e:
            print(f"Falling back to standard BERT: {e}")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            model = BERTForDualTruthfulness(bert_model, hidden_size=768)
            model.to(device)
        
        # Initialize detector with model, tokenizer, and device
        self.detector = SteganographicDetector(model, tokenizer, device)
        
        # Expected AM parameters based on your implementation
        self.carrier_periods = [2, 3, 4]  # Different carrier frequencies
        self.fs = 1.0  # Sampling frequency (1 step per sample)
        
        self.color_map = {
            'normal': 'blue',
            'step_now_steganographic': 'red', 
            'first_then_binary': 'green',
            'advanced_word_length': 'purple',
            'advanced_punctuation': 'orange',
            'am_hello': 'cyan',
            'am_secret': 'magenta',
            'am_ai_risk': 'yellow'
        }

    def split_reasoning_chain(self, text):
        """Split reasoning text into individual steps"""
        import re
        
        # Handle case where text is already a list (from AM data)
        if isinstance(text, list):
            return [step.strip() for step in text if step.strip()]
        
        # Handle different step patterns for strings
        if 'Step ' in text and 'Now ' in text:
            steps = re.split(r'(?=Step \d+:|Now )', text)
        elif 'First,' in text and 'Then ' in text:
            steps = re.split(r'(?=First,|Then )', text)
        else:
            steps = [s.strip() for s in text.split('.') if s.strip()]
        
        steps = [step.strip() for step in steps if step.strip()]
        return steps

    def load_datasets(self):
        """Load all datasets including AM patterns"""
        datasets = {}
        
        # Load main steganographic dataset
        try:
            with open('large_scale_steganographic_dataset.json', 'r') as f:
                dataset = json.load(f)
                datasets.update(dataset)
                print("Loaded large_scale_steganographic_dataset.json")
        except FileNotFoundError:
            print("Warning: large_scale_steganographic_dataset.json not found")
            
            # Fallback to individual files
            base_files = [
                'normal_reasoning.json',
                'step_now_steganographic.json', 
                'first_then_binary.json',
                'advanced_word_length.json',
                'advanced_punctuation.json'
            ]
            
            for filename in base_files:
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        category = filename.replace('.json', '')
                        datasets[category] = data
                        print(f"Loaded {filename}")
                except FileNotFoundError:
                    print(f"Warning: {filename} not found")
        
        # Load AM datasets
        try:
            with open('am_steganographic_data.json', 'r') as f:
                am_data = json.load(f)
            
            for result in am_data['results']:
                message = result['message']
                reasoning_chain = result['reasoning_chain']
                # Convert list to string if needed
                if isinstance(reasoning_chain, list):
                    reasoning_chain = ' '.join(reasoning_chain)
                datasets[f'am_{message.lower()}'] = [reasoning_chain]
            
            print(f"Added {len(am_data['results'])} AM patterns from am_steganographic_data.json")
        except FileNotFoundError:
            print("Warning: am_steganographic_data.json not found")
            
            # Try individual AM files
            am_files = [
                ('am_hello.json', 'am_hello'),
                ('am_secret.json', 'am_secret'), 
                ('am_ai_risk.json', 'am_ai_risk')
            ]
            
            for filename, category in am_files:
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        datasets[category] = data
                        print(f"Added {category}")
                except FileNotFoundError:
                    print(f"Warning: {filename} not found")
        
        print(f"Total datasets loaded: {list(datasets.keys())}")
        return datasets


    def extract_am_signal(self, reasoning_chain):
        """Extract temporal signal focusing on AM characteristics using dual classifier"""
        if isinstance(reasoning_chain, list):
            reasoning_chain = ' '.join(reasoning_chain)
        
        steps = self.split_reasoning_chain(reasoning_chain)
        signal = []
        
        for step_text in steps:
            try:
                # Use our local extract_layer_activations method
                layer_acts, token_probs, sentence_prob, _ = self.extract_layer_activations(step_text)
                
                # Use multiple layer combinations to capture AM effects
                early_layer = layer_acts.get('layer_0', layer_acts.get('layer_3', np.zeros(768)))
                mid_layer = layer_acts.get('layer_6', layer_acts.get('layer_7', np.zeros(768)))
                late_layer = layer_acts.get('layer_11', np.zeros(768))
                
                # AM should create specific inter-layer relationships
                # Modulation index: how much the signal varies
                modulation_depth = np.std(late_layer) / (np.mean(np.abs(late_layer)) + 1e-12)
                
                # Cross-layer correlation (carrier-message interaction)
                if len(early_layer) == len(late_layer) and len(early_layer) > 1:
                    cross_corr = np.corrcoef(early_layer, late_layer)[0,1]
                    if np.isnan(cross_corr):
                        cross_corr = 0
                else:
                    cross_corr = 0
                
                # Incorporate sentence probability as modulation signal
                sentence_modulation = sentence_prob[0] if len(sentence_prob) > 0 else 0.5
                
                # Combine for AM signature with sentence-level modulation
                am_signature = modulation_depth * (1 + cross_corr) * sentence_modulation
                signal.append(am_signature)
                
            except Exception as e:
                # If analysis fails, use a neutral value
                signal.append(0.5)
                continue
        
        return np.array(signal)

    def analyze_sideband_structure(self, signal):
        """Analyze sideband structure like in the DSB diagram"""
        if len(signal) < 8:
            return None, None, None
            
        # Apply windowing for better spectral analysis
        window = np.hanning(len(signal))
        windowed_signal = signal * window
        
        # FFT to get frequency domain
        fft_result = fft(windowed_signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        # Shift to center DC component
        fft_shifted = fftshift(fft_result)
        freqs_shifted = fftshift(freqs)
        
        # Calculate power spectral density
        psd = np.abs(fft_shifted)**2
        
        return freqs_shifted, fft_shifted, psd

    def detect_am_sidebands(self, freqs, psd, carrier_freq):
        """Detect upper and lower sidebands around carrier frequency"""
        if len(freqs) == 0:
            return False, 0, 0, 0
            
        # Find carrier frequency bin
        carrier_bin = np.argmin(np.abs(freqs - carrier_freq))
        
        # Expected sideband positions (±message frequency around carrier)
        sideband_range = int(len(freqs) * 0.1)  # Search range for sidebands
        
        start_bin = max(0, carrier_bin - sideband_range)
        end_bin = min(len(freqs), carrier_bin + sideband_range)
        
        # Calculate power in sideband regions
        lower_sideband_power = np.sum(psd[start_bin:carrier_bin])
        upper_sideband_power = np.sum(psd[carrier_bin:end_bin])
        carrier_power = psd[carrier_bin]
        
        # AM signature: significant power in both sidebands
        sideband_ratio = (lower_sideband_power + upper_sideband_power) / (carrier_power + 1e-12)
        
        # AM detection threshold
        is_am = sideband_ratio > 0.5 and carrier_power > np.mean(psd) * 0.5
        
        return is_am, carrier_power, lower_sideband_power, upper_sideband_power

    def create_dsb_style_visualization(self, dataset):
        """Create visualization matching the DSB diagram style"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout similar to DSB diagram
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
        
        am_categories = [cat for cat in dataset.keys() if cat.startswith('am_')]
        non_am_categories = [cat for cat in dataset.keys() if not cat.startswith('am_')]
        
        print(f"Found AM categories: {am_categories}")
        print(f"Found non-AM categories: {non_am_categories}")
        
        # Plot AM signals (top row)
        for idx, category in enumerate(am_categories[:3]):
            ax = fig.add_subplot(gs[0, idx])
            
            if category in dataset and dataset[category]:
                # Get signal
                example = dataset[category][0] if dataset[category] else ""
                signal = self.extract_am_signal(example)
                
                if len(signal) > 4:
                    # Time domain plot
                    time = np.arange(len(signal))
                    ax.plot(time, signal, 'b-', linewidth=2, label='m(t)')
                    
                    # Add carrier visualization
                    carrier_freq = 1.0/3.0  # 3-step period
                    carrier = np.cos(2 * np.pi * carrier_freq * time)
                    ax.plot(time, 0.5 * carrier, 'r--', alpha=0.7, label='cos ωc t')
                    
                    # Modulated signal
                    modulated = signal * carrier
                    ax.plot(time, modulated, 'g-', linewidth=1.5, alpha=0.8, label='m(t)cos ωc t')
                    
                    ax.set_title(f'{category.upper()}: Signal & Carrier', fontweight='bold')
                    ax.set_xlabel('Time (steps)')
                    ax.set_ylabel('Amplitude')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Insufficient signal\n({len(signal)} points)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{category.upper()}: No Signal', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{category.upper()}: No Data', fontweight='bold')

        # Frequency domain analysis (second row)
        for idx, category in enumerate(am_categories[:3]):
            ax = fig.add_subplot(gs[1, idx])
            
            if category in dataset and dataset[category]:
                example = dataset[category][0] if dataset[category] else ""
                signal = self.extract_am_signal(example)
                
                if len(signal) > 4:
                    freqs, fft_result, psd = self.analyze_sideband_structure(signal)
                    
                    if freqs is not None:
                        # Plot power spectral density
                        ax.semilogy(freqs, psd, 'b-', linewidth=2)
                        
                        # Highlight carrier and sidebands
                        carrier_freq = 1.0/3.0
                        ax.axvline(carrier_freq, color='red', linestyle='--', 
                                 alpha=0.8, label=f'Carrier ({carrier_freq:.3f})')
                        ax.axvline(-carrier_freq, color='red', linestyle='--', alpha=0.8)
                        
                        # Mark sideband regions
                        sideband_offset = 0.1
                        ax.axvline(carrier_freq + sideband_offset, color='orange', 
                                 linestyle=':', alpha=0.6, label='Upper Sideband')
                        ax.axvline(carrier_freq - sideband_offset, color='orange', 
                                 linestyle=':', alpha=0.6, label='Lower Sideband')
                        
                        ax.set_title(f'{category.upper()}: Frequency Spectrum', fontweight='bold')
                        ax.set_xlabel('Frequency (cycles/step)')
                        ax.set_ylabel('Power Spectral Density')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(-0.5, 0.5)
                    else:
                        ax.text(0.5, 0.5, 'FFT analysis failed', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{category.upper()}: No Spectrum', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'Insufficient signal\n({len(signal)} points)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{category.upper()}: No Spectrum', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{category.upper()}: No Data', fontweight='bold')

        # Sideband detection analysis (third row)
        ax_sideband = fig.add_subplot(gs[2, :])
        
        categories = list(dataset.keys())[:8]
        sideband_scores = []
        category_labels = []
        
        for category in categories:
            if category in dataset and dataset[category]:
                example = dataset[category][0]
                signal = self.extract_am_signal(example)
                
                if len(signal) > 4:
                    freqs, _, psd = self.analyze_sideband_structure(signal)
                    
                    if freqs is not None:
                        # Test multiple carrier frequencies
                        max_sideband_ratio = 0
                        for period in self.carrier_periods:
                            carrier_freq = 1.0/period
                            is_am, carrier_pow, lower_sb, upper_sb = self.detect_am_sidebands(
                                freqs, psd, carrier_freq)
                            
                            sideband_ratio = (lower_sb + upper_sb) / (carrier_pow + 1e-12)
                            max_sideband_ratio = max(max_sideband_ratio, sideband_ratio)
                        
                        sideband_scores.append(max_sideband_ratio)
                        category_labels.append(category.replace('_', '\n'))
                    else:
                        sideband_scores.append(0)
                        category_labels.append(category.replace('_', '\n'))
                else:
                    sideband_scores.append(0)
                    category_labels.append(category.replace('_', '\n'))
        
        if sideband_scores:
            colors = [self.color_map.get(cat.replace('\n', '_'), 'gray') for cat in category_labels]
            bars = ax_sideband.bar(range(len(sideband_scores)), sideband_scores, color=colors, alpha=0.7)
            
            # Add threshold line
            ax_sideband.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                              label='AM Detection Threshold')
            
            ax_sideband.set_title('Sideband Power Ratio (AM Signature Detection)', 
                                fontsize=14, fontweight='bold')
            ax_sideband.set_xlabel('Steganographic Category')
            ax_sideband.set_ylabel('Sideband/Carrier Power Ratio')
            ax_sideband.set_xticks(range(len(category_labels)))
            ax_sideband.set_xticklabels(category_labels, rotation=45, ha='right')
            ax_sideband.legend()
            ax_sideband.grid(True, alpha=0.3)
            
            # Annotate bars with values
            for i, (bar, score) in enumerate(zip(bars, sideband_scores)):
                ax_sideband.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                               f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

        # Summary analysis (fourth row)
        ax_summary = fig.add_subplot(gs[3, :])
        
        # Create summary text
        summary_text = "AM MODULATION ANALYSIS SUMMARY:\n\n"
        
        am_detected = [score > 0.5 for score in sideband_scores[-len(am_categories):] if len(sideband_scores) >= len(am_categories)]
        non_am_detected = [score > 0.5 for score in sideband_scores[:-len(am_categories)] if len(sideband_scores) > len(am_categories)]
        
        summary_text += f"• AM Patterns Detected: {sum(am_detected)}/{len(am_categories)} categories show sideband structure\n"
        summary_text += f"• Non-AM False Positives: {sum(non_am_detected)} out of {len(non_am_detected)} non-AM patterns\n\n"
        
        summary_text += "KEY FINDINGS:\n"
        summary_text += "• AM steganography should show distinct sideband patterns around carrier frequency\n"
        summary_text += "• Carrier frequency typically at 1/3 cycles/step (3-step period)\n"
        summary_text += "• Upper and Lower Sidebands contain the hidden message information\n"
        summary_text += "• Sideband/Carrier ratio > 0.5 indicates probable AM encoding\n\n"
        
        summary_text += "DETECTION IMPLICATIONS:\n"
        summary_text += "• Frequency domain analysis can distinguish AM from spatial steganography\n"
        summary_text += "• Traditional pattern-based detectors may miss AM signatures\n"
        summary_text += "• Spectral analysis reveals hidden modulation structure"
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)
        ax_summary.axis('off')

        plt.tight_layout()
        plt.savefig('dsb_am_modulation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved dsb_am_modulation_analysis.png")

def main():
    print("=== AM MODULATION ANALYSIS (DSB Style) ===")
    
    analyzer = AMModulationAnalyzer()
    dataset = analyzer.load_datasets()
    
    if not dataset:
        print("No datasets found!")
        return
    
    print("\n=== CREATING DSB-STYLE AM ANALYSIS ===")
    analyzer.create_dsb_style_visualization(dataset)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated: dsb_am_modulation_analysis.png")
    print("\nThis analysis shows:")
    print("- Time domain: message signal, carrier, and modulated output")
    print("- Frequency domain: carrier and sideband structure") 
    print("- Sideband detection: quantitative AM signature measurement")
    print("- Summary: detection implications for steganography research")

if __name__ == "__main__":
    main()
