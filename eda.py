import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from datetime import datetime

print("[INFO] Loading data from the database...")
# Load data from the database
conn = sqlite3.connect("detections.db")
df = pd.read_sql_query("SELECT * FROM detections", conn)
conn.close()

print("[INFO] Preprocessing data...")
# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Compute additional columns
df['minute'] = df['timestamp'].dt.floor('min')
df['area'] = df['width'] * df['height']

print("[INFO] Generating visualizations...")
# Create a multipage figure layout
sns.set(style="whitegrid")
fig, axs = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle("YOLOv8 Detection EDA Report", fontsize=20)

# Plot 1: Object Count
sns.countplot(data=df, x='object', order=df['object'].value_counts().index, ax=axs[0, 0])
axs[0, 0].set_title("Frequency of Detected Objects")
axs[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Confidence Over Time
sns.lineplot(data=df, x='timestamp', y='confidence', hue='object', marker='o', ax=axs[0, 1])
axs[0, 1].set_title("Detection Confidence Over Time")
axs[0, 1].tick_params(axis='x', rotation=30)

# Plot 3: Most Frequent Object Per Minute
most_common = df.groupby(['minute', 'object']).size().reset_index(name='count')
top_per_minute = most_common.sort_values(['minute', 'count'], ascending=[True, False])
top_per_minute = top_per_minute.groupby('minute').first().reset_index()
sns.countplot(data=top_per_minute, x='object', order=top_per_minute['object'].value_counts().index, ax=axs[1, 0])
axs[1, 0].set_title("Most Frequent Object Detected Per Minute")
axs[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Confidence vs Area
sns.scatterplot(data=df, x='confidence', y='area', hue='object', alpha=0.6, ax=axs[1, 1])
axs[1, 1].set_title("Confidence Score vs Bounding Box Area")

# Plot 5: Person Count Over Time
person_df = df[df['object'] == 'person']
if not person_df.empty:
    person_df.set_index('timestamp', inplace=True)
    person_df.resample('1Min')['object'].count().plot(ax=axs[2, 0])
    axs[2, 0].set_title("Number of Persons Detected Over Time")
    axs[2, 0].set_ylabel("Count")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].tick_params(axis='x', rotation=30)
else:
    axs[2, 0].text(0.3, 0.5, "No 'person' detected", fontsize=14)
    axs[2, 0].set_axis_off()

# Hide unused subplot
axs[2, 1].axis('off')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("[INFO] Saving EDA report and data...")
# Define path to save the PDF
output_dir = "C:/Users/siddh/Downloads/eda_report.pdf"
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

# Save the figure to the specified path
fig.savefig(output_dir)
print(f"[INFO] EDA report saved as {output_dir}")

# Save the processed data to CSV for Power BI
df.to_csv("detection_data.csv", index=False)
print("[INFO] CSV data exported for Power BI.")

# Show interactively
plt.show()

# Run Power BI Dashboard
print("[INFO] Launching Power BI dashboard...")
try:
    pbix_path = r"C:\Users\siddh\Desktop\eda major project\Detection_dashoard.pbix"
    subprocess.Popen(["start", "", pbix_path], shell=True)
    print("[INFO] Power BI dashboard launched.")
except Exception as e:
    print(f"[ERROR] Could not open Power BI dashboard: {e}")
