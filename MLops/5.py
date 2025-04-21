# 1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load Dataset (e.g., Titanic)
df = sns.load_dataset("titanic")

# 3. Basic Info
print(df.head())
print(df.info())
print(df.isnull().sum())

# 4. Data Cleaning
df.drop(['deck'], axis=1, inplace=True)  # too many nulls
df['age'].fillna(df['age'].median(), inplace=True)
df.dropna(subset=['embarked'], inplace=True)

# 5. Visualization
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.clf()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.clf()

sns.boxplot(x='survived', y='age', data=df)
plt.title("Age vs Survival")
plt.savefig("age_survival_boxplot.png")
plt.clf()

# 6. Generate Report
import pandas_profiling
profile = df.profile_report(title="Titanic EDA Report")
profile.to_file("titanic_eda_report.html")
