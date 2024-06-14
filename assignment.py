import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'boston_housing.csv'
df = pd.read_csv(file_path)

# Remove the Unnamed: 0 column if it exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Discretize the AGE variable into three groups
df['AGE_Group'] = pd.cut(df['AGE'], bins=[0, 35, 70, 100], labels=[
                         '35 years and younger', 'Between 35 and 70 years', '70 years and older'])

# TASK 2

# Boxplot for MEDV
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['MEDV'])
plt.title('Boxplot of Median Value of Owner-Occupied Homes')
plt.ylabel('Median Value (in $1000s)')
plt.show()

# Bar plot for CHAS
plt.figure(figsize=(10, 6))
sns.countplot(x='CHAS', data=df)
plt.title('Bar Plot of Charles River Variable')
plt.xlabel('Charles River (1 if tract bounds river; 0 otherwise)')
plt.ylabel('Frequency')
plt.show()

# Boxplot for MEDV vs AGE Group
plt.figure(figsize=(10, 6))
sns.boxplot(x='AGE_Group', y='MEDV', data=df)
plt.title('Boxplot of Median Value of Owner-Occupied Homes vs Age Group')
plt.xlabel('Age Group')
plt.ylabel('Median Value (in $1000s)')
plt.show()

# Scatter plot for NOX vs INDUS
plt.figure(figsize=(10, 6))
sns.scatterplot(x='INDUS', y='NOX', data=df)
plt.title('Scatter Plot of Nitric Oxide Concentrations vs Non-Retail Business Acres')
plt.xlabel('Proportion of Non-Retail Business Acres')
plt.ylabel('Nitric Oxide Concentrations (parts per 10 million)')
plt.show()

# Histogram for PTRATIO
plt.figure(figsize=(10, 6))
sns.histplot(df['PTRATIO'], bins=10, kde=True)
plt.title('Histogram of Pupil-Teacher Ratio')
plt.xlabel('Pupil-Teacher Ratio')
plt.ylabel('Frequency')
plt.show()

# TASK 3

# Load the dataset
file_path = 'boston_housing.csv'  # Adjust the path if necessary
df = pd.read_csv(file_path)

# Remove the Unnamed: 0 column if it exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Hypothesis 1: T-test for independent samples (CHAS)
# H0: There is no significant difference in median value of houses bounded by the Charles river or not.
# H1: There is a significant difference in median value of houses bounded by the Charles river or not.

chas_yes = df[df['CHAS'] == 1]['MEDV']
chas_no = df[df['CHAS'] == 0]['MEDV']

t_stat, p_value = stats.ttest_ind(chas_yes, chas_no)

print("T-test for independent samples (CHAS):")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis. There is a significant difference in median value of houses bounded by the Charles river or not.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no significant difference in median value of houses bounded by the Charles river or not.")
#conslusion
#p-value < 0.05 -> H0 rejected

# Hypothesis 2: ANOVA (AGE)
# H0: There is no difference in Median values of houses for each proportion of owner occupied units built prior to 1940.
# H1: There is a difference in Median values of houses for each proportion of owner occupied units built prior to 1940.

df['AGE_Group'] = pd.cut(df['AGE'], bins=[0, 35, 70, 100], labels=[
                         '35 years and younger', 'Between 35 and 70 years', '70 years and older'])

anova_result = stats.f_oneway(
    df[df['AGE_Group'] == '35 years and younger']['MEDV'],
    df[df['AGE_Group'] == 'Between 35 and 70 years']['MEDV'],
    df[df['AGE_Group'] == '70 years and older']['MEDV']
)

print("\nANOVA (AGE):")
print(f"F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")

if anova_result.pvalue < 0.05:
    print("Conclusion: Reject the null hypothesis. There is a difference in Median values of houses for each proportion of owner occupied units built prior to 1940.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no difference in Median values of houses for each proportion of owner occupied units built prior to 1940.")
# Conclusion 
#p-value < 0.05 -> H0 rejected

# Hypothesis 3: Pearson Correlation (NOX and INDUS)
# H0: There is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.
# H1: There is a relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.

correlation, p_value = stats.pearsonr(df['NOX'], df['INDUS'])

print("\nPearson Correlation (NOX and INDUS):")
print(f"Correlation coefficient: {correlation}, P-value: {p_value}")

if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis. There is a relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.")
# conslusion
# p-value <<0.05 -> reject H0

# Hypothesis 4: Regression analysis (DIS and MEDV)
# H0: There is no impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes.
# H1: There is an impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes.

X = df['DIS']
y = df['MEDV']
X = sm.add_constant(X)  # Add constant term for the intercept

model = sm.OLS(y, X).fit()

print("\nRegression analysis (DIS and MEDV):")
print(model.summary())

if model.pvalues[1] < 0.05:
    print("Conclusion: Reject the null hypothesis. There is an impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes.")
