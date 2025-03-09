import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
file_path = '/Users/jashanshetty29/newp/expandedanhanced1.csv'
data = pd.read_csv(file_path)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.columns = data.columns.str.strip()
data['Chance of Admit'] = data['Chance of Admit'] * 100

# Generate correlation matrix
correlation_matrix = data.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
