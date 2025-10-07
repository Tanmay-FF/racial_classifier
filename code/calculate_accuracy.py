import pandas as pd

# Path to your CSV file
csv_path = r"sample_result.csv"  # <-- change this path if needed

# Read the CSV file
df = pd.read_csv(csv_path)

# Compute per-class accuracy
class_accuracies = {}
for cls in df['original_label'].unique():
    cls_df = df[df['original_label'] == cls]
    correct = (cls_df['original_label'] == cls_df['predicted_label']).sum()
    total = len(cls_df)
    accuracy = (correct / total) * 100 if total > 0 else 0
    class_accuracies[cls] = round(accuracy, 2)

# Convert to a nice DataFrame for display
result_df = pd.DataFrame(list(class_accuracies.items()), columns=['Class', 'Accuracy (%)'])
print(result_df)

# Optionally save to CSV
result_df.to_csv("class_wise_accuracy.csv", index=False)
