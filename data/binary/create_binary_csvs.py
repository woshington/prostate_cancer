import pandas as pd
import os

# Read the original CSV
df = pd.read_csv('../train_5_fold.csv')

print(f"Original dataset shape: {df.shape}")
print(f"\nISUP grade distribution:")
print(df['isup_grade'].value_counts().sort_index())

# Create binary CSV for each ISUP grade (0-5)
for grade in range(6):
    # Create binary label: 1 if isup_grade == grade, 0 otherwise
    binary_df = df.copy()
    binary_df['binary_label'] = (binary_df['isup_grade'] == grade).astype(int)

    # Save to CSV
    output_file = f'isup_grade_{grade}_binary.csv'
    binary_df.to_csv(output_file, index=False)

    # Print statistics
    positive_samples = binary_df['binary_label'].sum()
    negative_samples = len(binary_df) - positive_samples
    print(f"\nISUP Grade {grade} binary classification:")
    print(f"  File: {output_file}")
    print(f"  Positive samples (grade={grade}): {positive_samples}")
    print(f"  Negative samples (grade!={grade}): {negative_samples}")
    print(f"  Imbalance ratio: 1:{negative_samples/positive_samples:.2f}")

print("\n✓ All binary CSV files created successfully!")