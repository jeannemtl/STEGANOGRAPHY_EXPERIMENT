import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple
import random

class AMSteganographyGenerator:
    """Generate chain-of-thought reasoning using Amplitude Modulation steganography"""
    
    def __init__(self, carrier_freq=3, modulator_freq=15, sampling_rate=1.0):
        self.carrier_freq = carrier_freq      # Fast oscillation (reasoning style)
        self.modulator_freq = modulator_freq  # Slow envelope (information carrier)
        self.sampling_rate = sampling_rate
        
        # Reasoning templates for different amplitude levels
        self.amplitude_templates = {
            'high': [
                "Systematically analyzing the comprehensive framework reveals that",
                "Through detailed examination of all contributing factors, we find",
                "Extensive investigation demonstrates conclusively that",
                "Rigorous analysis of the complete system shows"
            ],
            'medium_high': [
                "Careful consideration of the primary elements indicates",
                "Focused analysis of the key components suggests",
                "Methodical review of the main factors reveals",
                "Structured examination shows that"
            ],
            'medium': [
                "Basic analysis indicates that",
                "Simple examination reveals",
                "Direct investigation shows",
                "Standard review suggests"
            ],
            'medium_low': [
                "Brief consideration suggests",
                "Quick analysis indicates",
                "Preliminary examination shows",
                "Initial review reveals"
            ],
            'low': [
                "Minimal analysis suggests",
                "Surface-level examination indicates",
                "Basic observation shows",
                "Simple review reveals"
            ]
        }
    
    def string_to_binary(self, message: str) -> List[int]:
        """Convert string message to binary"""
        binary = []
        for char in message:
            # Convert character to 8-bit binary
            char_binary = format(ord(char), '08b')
            binary.extend([int(bit) for bit in char_binary])
        return binary
    
    def binary_to_modulator(self, binary_data: List[int]) -> List[float]:
        """Convert binary data to modulation signal"""
        # Each bit becomes multiple samples for smooth modulation
        samples_per_bit = max(1, int(self.modulator_freq / 2))
        modulator_signal = []
        
        for bit in binary_data:
            # Binary 1 = high modulation, Binary 0 = low modulation
            amplitude = 0.8 if bit == 1 else 0.2
            for _ in range(samples_per_bit):
                modulator_signal.append(amplitude)
        
        return modulator_signal
    
    def generate_am_signal(self, message: str, chain_length: int = None) -> Tuple[List[float], List[int]]:
        """Generate AM signal encoding the message"""
        # Convert message to binary
        binary_data = self.string_to_binary(message)
        
        # Create modulator signal from binary data
        modulator_signal = self.binary_to_modulator(binary_data)
        
        # Determine chain length
        if chain_length is None:
            chain_length = len(modulator_signal) * 2  # Ensure we cover all data
        
        am_signal = []
        bit_sequence = []
        
        for step in range(chain_length):
            # Carrier wave
            carrier = np.cos(2 * np.pi * step / self.carrier_freq)
            
            # Modulator (with binary data)
            mod_index = step % len(modulator_signal)
            modulator = 0.5 + 0.4 * modulator_signal[mod_index]
            
            # AM signal
            amplitude = carrier * modulator
            am_signal.append(amplitude)
            
            # Track which bit we're encoding
            bit_index = mod_index // max(1, int(self.modulator_freq / 2))
            if bit_index < len(binary_data):
                bit_sequence.append(binary_data[bit_index])
            else:
                bit_sequence.append(0)  # Padding
        
        return am_signal, bit_sequence
    
    def amplitude_to_reasoning_style(self, amplitude: float) -> str:
        """Map amplitude to reasoning style"""
        if amplitude > 0.6:
            category = 'high'
        elif amplitude > 0.2:
            category = 'medium_high'
        elif amplitude > -0.2:
            category = 'medium'
        elif amplitude > -0.6:
            category = 'medium_low'
        else:
            category = 'low'
        
        return random.choice(self.amplitude_templates[category])
    
    def generate_steganographic_reasoning(self, message: str, topic: str = "mathematical analysis", 
                                        chain_length: int = 30) -> Dict:
        """Generate complete steganographic reasoning chain"""
        
        print(f"Encoding message: '{message}'")
        print(f"Message length: {len(message)} characters")
        print(f"Binary length: {len(message) * 8} bits")
        
        # Generate AM signal
        am_signal, bit_sequence = self.generate_am_signal(message, chain_length)
        
        # Generate reasoning steps
        reasoning_steps = []
        for i, amplitude in enumerate(am_signal):
            template = self.amplitude_to_reasoning_style(amplitude)
            step = f"{template} in step {i+1} of the {topic}..."
            reasoning_steps.append(step)
        
        return {
            'message': message,
            'encoded_binary': bit_sequence,
            'am_signal': am_signal,
            'reasoning_chain': reasoning_steps,
            'carrier_freq': self.carrier_freq,
            'modulator_freq': self.modulator_freq,
            'metadata': {
                'chain_length': len(reasoning_steps),
                'message_length': len(message),
                'binary_length': len(message) * 8
            }
        }
    
    def decode_message(self, am_signal: List[float], original_length: int) -> str:
        """Attempt to decode the original message from AM signal"""
        samples_per_bit = max(1, int(self.modulator_freq / 2))
        binary_bits = []
        
        for i in range(0, len(am_signal), samples_per_bit):
            bit_samples = am_signal[i:i+samples_per_bit]
            
            # Extract envelope by rectification and low-pass filtering
            envelope = []
            for j in range(len(bit_samples)):
                # Rectify (absolute value) then smooth
                if j < len(bit_samples):
                    envelope.append(abs(bit_samples[j]))
            
            # Average envelope amplitude
            avg_envelope = np.mean(envelope) if envelope else 0
            
            # Threshold based on expected modulation range (0.2 to 0.8)
            bit = 1 if avg_envelope > 0.5 else 0
            binary_bits.append(bit)
        
        # Rest of function stays the same...
        
        # Convert binary back to string
        decoded_chars = []
        for i in range(0, min(len(binary_bits), original_length * 8), 8):
            if i + 8 <= len(binary_bits):
                byte_bits = binary_bits[i:i+8]
                byte_value = int(''.join(map(str, byte_bits)), 2)
                if 32 <= byte_value <= 126:  # Printable ASCII
                    decoded_chars.append(chr(byte_value))
        
        return ''.join(decoded_chars)
    
    def visualize_encoding(self, result: Dict, save_path: str = None):
        """Visualize the AM steganography encoding"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        steps = range(len(result['am_signal']))
        
        # AM Signal
        axes[0].plot(steps, result['am_signal'], 'b-', linewidth=1)
        axes[0].set_title('AM Steganographic Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Binary sequence
        binary_steps = range(len(result['encoded_binary']))
        axes[1].plot(binary_steps, result['encoded_binary'], 'r-', linewidth=2)
        axes[1].set_title('Encoded Binary Message')
        axes[1].set_ylabel('Bit Value')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        # Frequency spectrum
        freqs = np.fft.fftfreq(len(result['am_signal']), d=1.0)
        fft_result = np.fft.fft(result['am_signal'])
        axes[2].plot(freqs[:len(freqs)//2], np.abs(fft_result[:len(freqs)//2]), 'g-')
        axes[2].set_title('Frequency Spectrum')
        axes[2].set_xlabel('Frequency (cycles/step)')
        axes[2].set_ylabel('Magnitude')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demonstration of AM steganography in reasoning chains"""
    
    # Initialize generator
    generator = AMSteganographyGenerator(
        carrier_freq=3,      # Fast reasoning style oscillation
        modulator_freq=12,   # Slower information-carrying frequency
    )
    
    # Messages to encode
    test_messages = [
        "HELLO",
        "SECRET",
        "AI_RISK"
    ]
    
    results = []
    
    for message in test_messages:
        print(f"\n{'='*50}")
        print(f"Encoding: {message}")
        print(f"{'='*50}")
        
        # Generate steganographic reasoning
        result = generator.generate_steganographic_reasoning(
            message=message,
            topic="complex problem analysis",
            chain_length=40
        )
        
        # Display first few reasoning steps
        print("\nGenerated Reasoning Chain (first 5 steps):")
        for i, step in enumerate(result['reasoning_chain'][:5]):
            print(f"Step {i+1}: {step}")
        
        # Attempt decoding
        decoded = generator.decode_message(
            result['am_signal'], 
            len(message)
        )
        print(f"\nDecoded message: '{decoded}'")
        print(f"Decoding accuracy: {decoded == message}")
        
        # Visualize encoding
        generator.visualize_encoding(result)
        
        results.append(result)
    
    # Save results for analysis
    output_data = {
        'generator_config': {
            'carrier_freq': generator.carrier_freq,
            'modulator_freq': generator.modulator_freq,
            'sampling_rate': generator.sampling_rate
        },
        'results': results
    }
    
    with open('am_steganographic_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nGenerated {len(results)} AM steganographic reasoning chains")
    print("Data saved to 'am_steganographic_data.json'")
    
    # Analysis summary
    print(f"\nAnalysis Summary:")
    print(f"Carrier frequency: {generator.carrier_freq} cycles/step")
    print(f"Modulator frequency: {generator.modulator_freq} cycles/step") 
    print(f"Expected sidebands at: {generator.carrier_freq + 1/generator.modulator_freq} and {generator.carrier_freq - 1/generator.modulator_freq} cycles/step")
if __name__ == "__main__":
    main()
