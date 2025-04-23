import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Load Dataset

df = pd.read_csv("C:\Users\MoxDev\Desktop\EDA Analysis\wb_gender.csv)
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Data Cleaning

# Drop unnecessary columns
drop_cols = [col for col in df.columns if 'Unnamed' in col or col in ['id', 'iso2Code']]
df.drop(columns=drop_cols, inplace=True)

# Convert 'Year' to integer
df['Year'] = df['Year'].astype(int)

# Drop rows with missing Country or Year
df.dropna(subset=['Country', 'Year'], inplace=True)

# Drop columns with >40% missing values
df = df.loc[:, df.isnull().mean() < 0.4]

# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("After cleaning shape:", df.shape)

# 3. Summary Statistics

print("Data Description:\n", df.describe(include='all'))
print("Missing Values:\n", df.isnull().sum().sort_values(ascending=False).head())

# 4. Global Trends Over Time

# Group by Year and calculate global averages
global_trend = df.groupby('Year').mean().reset_index()

# Plot: Female Life Expectancy over time
plt.plot(global_trend['Year'], global_trend['Life expectancy at birth, female (years)'], marker='o')
plt.title("Global Female Life Expectancy Over Time")
plt.xlabel("Year")
plt.ylabel("Life Expectancy (Years)")
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Gender Comparison Over Time

plt.plot(global_trend['Year'], global_trend['Life expectancy at birth, female (years)'], label='Female', marker='o')
plt.plot(global_trend['Year'], global_trend['Life expectancy at birth, male (years)'], label='Male', marker='s')
plt.title("Life Expectancy by Gender")
plt.xlabel("Year")
plt.ylabel("Years")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Heatmap: Correlation Matrix

plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Development Indicators")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 7. Regional Insights (if Region column exists)
# ---------------------------------------------
if "Region" in df.columns:
    region_group = df.groupby("Region")[["Life expectancy at birth, female (years)",
                                         "GDP per capita (constant 2010 US$)",
                                         "Fertility rate, total (births per woman)"]].mean()
    print("Regional Averages:\n", region_group)

    # Boxplot: Female Life Expectancy by Region
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Region", y="Life expectancy at birth, female (years)")
    plt.title("Life Expectancy by Region")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# 8. Income Group Analysis (if IncomeGroup exists)
# ---------------------------------------------
if "IncomeGroup" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="IncomeGroup", y="GDP per capita (constant 2010 US$)", palette="Set2")
    plt.title("GDP per Capita by Income Group")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# 9. Internet Access Trends
# ---------------------------------------------
if "Individuals using the Internet (% of population)" in df.columns:
    internet_trend = df.groupby("Year")["Individuals using the Internet (% of population)"].mean().reset_index()
    plt.plot(internet_trend["Year"], internet_trend["Individuals using the Internet (% of population)"], marker='o')
    plt.title("Global Internet Access Over Time")
    plt.ylabel("Internet Users (%)")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# 10. Scatter Plot: Fertility vs Life Expectancy
# ---------------------------------------------
plt.scatter(df["Fertility rate, total (births per woman)"],
            df["Life expectancy at birth, female (years)"],
            c='mediumvioletred', alpha=0.6)
plt.title("Fertility Rate vs Life Expectancy (Female)")
plt.xlabel("Fertility Rate")
plt.ylabel("Life Expectancy (Years)")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 11. Distribution Plots
# ---------------------------------------------
# Histogram of GDP per capita
sns.histplot(data=df, x="GDP per capita (constant 2010 US$)", bins=30, kde=True, color="steelblue")
plt.title("Distribution of GDP per Capita")
plt.tight_layout()
plt.show()

# KDE Plot: Literacy Rate (if exists)
if "Literacy rate, adult female (% of females ages 15 and above)" in df.columns:
    sns.kdeplot(data=df, x="Literacy rate, adult female (% of females ages 15 and above)", fill=True, color="orange")
    plt.title("Literacy Rate Distribution (Adult Females)")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# 12. Pairplot for Key Indicators
# ---------------------------------------------
selected_cols = [
    "Life expectancy at birth, female (years)",
    "Life expectancy at birth, male (years)",
    "Fertility rate, total (births per woman)",
    "GDP per capita (constant 2010 US$)"
]
sns.pairplot(df[selected_cols].dropna(), corner=True, diag_kind="kde")
plt.suptitle("Pairplot of Key Gender Development Indicators", y=1.02)
plt.tight_layout()
plt.show()

df.to_csv("gender_cleaned.csv", index=False)
