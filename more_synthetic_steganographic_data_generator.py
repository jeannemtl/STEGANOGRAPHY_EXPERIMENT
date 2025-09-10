from steganographic_dataset_generator import SteganographicDatasetGenerator

generator = SteganographicDatasetGenerator()

# Generate much larger dataset for statistical power
dataset = {
    'normal': generator.generate_normal_reasoning(num_examples=200),
    'step_now_steganographic': generator.generate_step_now_steganographic(num_examples=200),
    'first_then_binary': generator.generate_first_then_binary(num_examples=200),
    'advanced_word_length': generator.generate_advanced_steganographic(num_examples=150, encoding_type='word_length'),
    'advanced_punctuation': generator.generate_advanced_steganographic(num_examples=150, encoding_type='punctuation')
}

# Save the larger dataset
generator.save_dataset(dataset, 'large_scale_steganographic_dataset.json')
