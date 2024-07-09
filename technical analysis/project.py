#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dash as dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from pandas_datareader import data
from datetime import date
from numpy.linalg import cond, svd
from scipy import stats

# %%
file_path = "Airlines_train.csv"
df = pd.read_csv(file_path)

#%% 
print(df.describe())

#%%
print(df.isnull().sum())

df.dropna(inplace=True)

print(df.isnull().sum())

#%%
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#%%
df = df.drop(columns=['X', 'id'])

#%%
df['Age.cat'] = None

df.loc[df['Age'] <= 20, 'Age.cat'] = 'Under 20'
df.loc[(df['Age'] >= 21) & (df['Age'] <= 40), 'Age.cat'] = '20-40'
df.loc[(df['Age'] >= 41) & (df['Age'] <= 80), 'Age.cat'] = '40-80'
df.loc[df['Age'] >= 80, 'Age.cat'] = 'above 80'

df['Age.cat'] = pd.Categorical(df['Age.cat'], categories=['Under 20', '20-40', '40-80', 'above 80'])

df['Age.cat2'] = None

df.loc[df['Age'] <= 20, 'Age.cat2'] = 1
df.loc[(df['Age'] >= 21) & (df['Age'] <= 40), 'Age.cat2'] = 2
df.loc[(df['Age'] >= 41) & (df['Age'] <= 80), 'Age.cat2'] = 3
df.loc[df['Age'] >= 80, 'Age.cat2'] = 4

print(df)

#%%

df['satisfaction_numeric'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)


#%%
df['satisfaction'] = df['satisfaction'].astype('category')
df['Class'] = df['Class'].astype('category')
df['Type of Travel'] = df['Type of Travel'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Customer Type'] = df['Customer Type'].astype('category')
df['Age.cat']=df['Age.cat'].astype('category')
df['Age.cat2']=df['Age.cat2'].astype('int64')
#%%
print(df.dtypes)

# %%
q1_Flight_Distance = df['Flight Distance'].quantile(0.25)
q3_Flight_Distance = df['Flight Distance'].quantile(0.75)
iqr_Flight_Distance= q3_Flight_Distance - q1_Flight_Distance
lower_bound = q1_Flight_Distance - 1.5 * iqr_Flight_Distance
upper_bound = q3_Flight_Distance + 1.5 * iqr_Flight_Distance

print(f"Q1 and Q3 of the Flight Distance is {q1_Flight_Distance:.2f}  & {q3_Flight_Distance:.2f} .")
print(f"IQR for the Flight Distance is {iqr_Flight_Distance:.2f} .")
print(f"Any Flight Distance < {lower_bound:.2f}  and Flight Distance > {upper_bound:.2f}  is an outlier.")

plt.figure(figsize=(8, 6))
plt.boxplot(df['Flight Distance'])
plt.xlabel('Flight Distance')
plt.title('Boxplot of Flight Distance')

plt.show()

cleaned_df = df[(df['Flight Distance'] >= lower_bound) & (df['Flight Distance'] <= upper_bound)]

plt.figure(figsize=(10, 6))
plt.boxplot(cleaned_df['Flight Distance'])
plt.title('Boxplot-Flight Distance-Cleaned')
plt.xlabel('Flight Distance')
plt.grid(True)
plt.show()

#%%
q1_Departure_Delay = df['Departure Delay in Minutes'].quantile(0.25)
q3_Departure_Delay= df['Departure Delay in Minutes'].quantile(0.75)
iqr_Departure_Delay = q3_Departure_Delay - q1_Departure_Delay
lower_bound = q1_Departure_Delay - 1.5 * iqr_Departure_Delay
upper_bound = q3_Departure_Delay + 1.5 * iqr_Departure_Delay

print(f"Q1 and Q3 of the Departure Delay is {q1_Departure_Delay:.2f}  & {q3_Departure_Delay:.2f} .")
print(f"IQR for the Departure Delay is {iqr_Departure_Delay:.2f} .")
print(f"Any Departure_Delay < {lower_bound:.2f}  and Departure_Delay > {upper_bound:.2f}  is an outlier.")

plt.figure(figsize=(8, 6))
plt.boxplot(df['Departure Delay in Minutes'])
plt.xlabel('Departure_Delay')
plt.title('Boxplot of Departure_Delay')

plt.show()

cleaned_df = df[(df['Departure Delay in Minutes'] >= lower_bound) & (df['Departure Delay in Minutes'] <= upper_bound)]

plt.figure(figsize=(10, 6))
plt.boxplot(cleaned_df['Departure Delay in Minutes'])
plt.title('Boxplot-Departure_Delay-Cleaned')
plt.xlabel('Departure_Delay')
plt.grid(True)
plt.show()

#%%
q1_Arrival_Delay = df['Arrival Delay in Minutes'].quantile(0.25)
q3_Arrival_Delay= df['Arrival Delay in Minutes'].quantile(0.75)
iqr_Arrival_Delay = q3_Arrival_Delay - q1_Arrival_Delay
lower_bound = q1_Arrival_Delay - 1.5 * iqr_Arrival_Delay
upper_bound = q3_Arrival_Delay + 1.5 * iqr_Arrival_Delay

print(f"Q1 and Q3 of the Arrival_Delay is {q1_Departure_Delay:.2f}  & {q3_Departure_Delay:.2f} .")
print(f"IQR for the Arrival_Delay is {iqr_Departure_Delay:.2f} .")
print(f"Any Arrival_Delay < {lower_bound:.2f}  and Arrival_Delay > {upper_bound:.2f}  is an outlier.")

plt.figure(figsize=(8, 6))
plt.boxplot(df['Arrival Delay in Minutes'])
plt.xlabel('Arrival_Delay')
plt.title('Boxplot of Arrival_Delay')

plt.show()

cleaned_df = df[(df['Arrival Delay in Minutes'] >= lower_bound) & (df['Arrival Delay in Minutes'] <= upper_bound)]

plt.figure(figsize=(10, 6))
plt.boxplot(cleaned_df['Arrival Delay in Minutes'])
plt.title('Boxplot-Arrival_Delay-Cleaned')
plt.xlabel('Arrival_Delay')
plt.grid(True)
plt.show()
#%%
print(df.head(5))
print(df.describe())
sns.set_style('whitegrid')

#%%
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='Age', alpha=0.6, linewidth=2, fill=True)
plt.title('Passenger Age Distribution')
plt.xlabel('Age')
plt.ylabel('Density of the Customers')
plt.show()
# %%
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

colors = {'Male': '#F06292', 'Female': '#9FA8DA', 'Other': 'gray'}

plt.bar(gender_counts['Gender'], gender_counts['Count'], color=[colors[gender] for gender in gender_counts['Gender']])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# %%
columns = ['Type of Travel', 'Class', 'Customer Type', 'satisfaction']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

for col, ax in zip(columns, axes.flatten()):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f'Count Plot of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
color_palette = {'satisfied': '#F06292', 'neutral or dissatisfied': '#9FA8DA', 'Other': 'gray'}

plt.figure(figsize=(8, 6))
df.groupby(['Type of Travel', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()])
plt.title('Customer Satisfaction according to Type of Travel')
plt.xlabel('Type of Travel')
plt.ylabel('Customer Volume')
plt.legend(title='Satisfaction', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.show()
# %%
plt.figure(figsize=(8, 6))
df.groupby(['Class', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()])
plt.title('Customer Satisfaction based on Class')
plt.xlabel('Class')
plt.ylabel('Number of Customers')
plt.legend(title='Satisfaction', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.show()
# %%
plt.figure(figsize=(8, 6))
df.groupby(['Customer Type', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()])
plt.title('Customer Satisfaction according to Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Number of Customers')
plt.legend(title='Satisfaction', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.show()

#%%

color_palette = {'satisfied': '#F06292', 'neutral or dissatisfied': '#9FA8DA', 'Other': 'gray'}

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

df.groupby(['Type of Travel', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()], ax=axs[0])
axs[0].set_title('Customer Satisfaction according to Type of Travel')
axs[0].set_xlabel('Type of Travel')
axs[0].set_ylabel('Customer Volume')
axs[0].legend(title='Satisfaction', bbox_to_anchor=(1, 1))
axs[0].tick_params(axis='x', rotation=0)

df.groupby(['Class', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()], ax=axs[1])
axs[1].set_title('Customer Satisfaction based on Class')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('Number of Customers')
axs[1].legend(title='Satisfaction', bbox_to_anchor=(1, 1))
axs[1].tick_params(axis='x', rotation=0)

df.groupby(['Customer Type', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True, color=[color_palette.get(x, '#9E9E9E') for x in df['satisfaction'].unique()], ax=axs[2])
axs[2].set_title('Customer Satisfaction according to Customer Type')
axs[2].set_xlabel('Customer Type')
axs[2].set_ylabel('Number of Customers')
axs[2].legend(title='Satisfaction', bbox_to_anchor=(1, 1))
axs[2].tick_params(axis='x', rotation=0)

plt.tight_layout()  
plt.show()


# %%
class_gender_counts = df.groupby(['Class', 'Gender']).size().unstack()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create subplots for each class

for i, (class_name, class_data) in enumerate(class_gender_counts.items()):
    ax = axes[i]  
    ax.pie(class_data, labels=class_data.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'Distribution of Gender in {class_name} Class')
    ax.axis('equal')  
plt.tight_layout()

plt.show()

# %%
class_travel_counts = df.groupby(['Class', 'Type of Travel']).size().unstack()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for i, (class_name, class_data) in enumerate(class_travel_counts.items()):
    ax = axes[i] if len(class_travel_counts) > 1 else axes  # Use single axis if only one 'Class' category
    ax.pie(class_data, labels=class_data.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'Distribution of Type of Travel in {class_name} Class')
    ax.axis('equal') 
plt.tight_layout()

plt.show()
# %%

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Flight Distance', hue='Type of Travel', kde=True, bins=30, stat="count", multiple='stack')
plt.title('Distribution of Flight Distance by Number of Customers (with Type of Travel)')
plt.xlabel('Flight Distance')
plt.ylabel('Number of Occurrences')
plt.legend(title='Type of Travel', labels=['Personal Travel', 'Business Travel'])

plt.show()


# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Class', y='Seat comfort')
plt.title('Seat Comfort Distribution by Class (Box Plot)')
plt.xlabel('Class')
plt.ylabel('Seat Comfort')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Class', y='Seat comfort')
plt.title('Seat Comfort Distribution by Class (Violin Plot)')
plt.xlabel('Class')
plt.ylabel('Seat Comfort')
plt.show()

# %%
aspects = ['Inflight wifi service', 'Cleanliness', 'Food and drink', 'Checkin service', 'Seat comfort', 'Leg room service']
titles = ['In-flight WiFi Service', 'Cleanliness', 'Food and Drinks', 'Check-in Service', 'Seat Comfort', 'Leg room service']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

axes = axes.flatten()

for i, aspect in enumerate(aspects):
    sns.violinplot(ax=axes[i], data=df, x='satisfaction', y=aspect, hue='satisfaction')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Overall Satisfaction')
    axes[i].set_ylabel(aspect)
    axes[i].legend(title='Satisfaction')

plt.tight_layout()
plt.show()
# %%

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Customer Type', hue='satisfaction', palette='Set2')
plt.title('Count of Checkin Service Satisfaction by Type of Customer')
plt.xlabel('Type of Customer')
plt.ylabel('Count')
plt.legend(title='Satisfaction', bbox_to_anchor=(1, 1))

plt.show()


# %%
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Type of Travel', hue='satisfaction')
plt.title('Count Plot: Type of Travel')
plt.xlabel('Type of Travel')
plt.ylabel('Count')
plt.legend(title='Satisfaction')
plt.show()
# %%

selected_columns = ['Age', 'Gender', 'Flight Distance']
filtered_df = df[selected_columns]

g = sns.jointplot(data=filtered_df, x='Age', y='Flight Distance', hue='Gender', kind='scatter', palette='Set2')
plt.title('Joint Plot: Age vs Flight Distance with Gender Hue')
plt.show()


# %%
grouped_data = df.groupby(['Gender', 'satisfaction']).size().unstack()

grouped_data_norm = grouped_data.div(grouped_data.sum(axis=1), axis=0)

plt.figure(figsize=(8, 6))
grouped_data.plot(kind='bar', stacked=True, color=['#F06292', '#9FA8DA'])
plt.title('Gender vs Satisfaction')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Satisfaction')

plt.show()

# %%

columns = ['Age', 'Flight Distance']

fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 6 * len(columns)))

for idx, col in enumerate(columns):
    count_data = df[col].value_counts().reset_index()
    count_data.columns = [col, 'Count']

    sns.kdeplot(data=count_data, x=col, y='Count', fill=True, alpha=0.6, ax=axes[idx], linewidth=1)
    axes[idx].set_title(f'{col} Distribution by Count')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count Density')

plt.tight_layout()
plt.show()



# %%
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Age', y='Flight Distance', hue='satisfaction', linewidth=3)
plt.title('Line Plot: Age vs Flight Distance')
plt.xlabel('Age')
plt.ylabel('Flight Distance')
plt.legend(title='Satisfaction')
plt.show()

#%%
plt.figure(figsize=(8, 6))
stats.probplot(df['Flight Distance'], dist="norm", plot=plt)
plt.title('QQ Plot: Flight Distance')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Flight Distance')
plt.show()

#%%
plt.figure(figsize=(8, 6))
stats.probplot(df['Departure Delay in Minutes'], dist="norm", plot=plt)
plt.title('QQ Plot: Departure Delay in Minutes')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Departure Delay in Minutes')
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Online boarding', 'Gate location', 'Ease of Online booking', 'Baggage handling']], orient='h')
plt.title('Multivariate Box Plot')
plt.xlabel('Rating')
plt.ylabel('Features')
plt.show()

# %%

features = ['Online boarding', 'Gate location', 'Ease of Online booking', 'Baggage handling', 'satisfaction']

melted_df = df[features].melt(id_vars=['satisfaction'], var_name='Feature', value_name='Rating')

plt.figure(figsize=(10, 6))
sns.violinplot(data=melted_df, x='Rating', y='Feature', hue='satisfaction', split=True)
plt.title('Violin Plot with Satisfaction')
plt.xlabel('Rating')
plt.ylabel('Features')
plt.legend(title='Satisfaction', loc='lower right')
plt.show()


# %%
plt.figure(figsize=(10, 6))
df[['Inflight wifi service', 'Cleanliness']].plot.area(stacked=False, alpha=0.5)
plt.title('Area Plot: Satisfaction Ratings')
plt.xlabel('Observations')
plt.ylabel('Rating')
plt.legend(title='Feature' )
plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['Age'], df['Flight Distance'], df['Seat comfort'], c=df['satisfaction'].cat.codes, cmap='viridis')

legend_labels = df['satisfaction'].cat.categories.tolist()
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(level)), label=label) for level, label in enumerate(legend_labels)]
ax.legend(handles=legend_handles, title='satisfaction')

ax.set_xlabel('Age')
ax.set_ylabel('Flight Distance')
ax.set_zlabel('Seat Comfort')
plt.title('3D Scatter Plot: Age vs Flight Distance vs Seat Comfort')

plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.hexbin(x=df['Age'], y=df['Flight Distance'], gridsize=30, cmap='inferno')
plt.colorbar(label='count in bin')
plt.title('Hexbin Plot: Age vs Flight Distance')
plt.xlabel('Age')
plt.ylabel('Flight Distance')
plt.show()
# %%
custom_palette = {'satisfied': 'green', 'neutral or dissatisfied': 'orange', 'Other': 'gray'}
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='Type of Travel', y='Flight Distance', hue='satisfaction', palette=custom_palette, jitter=True)
plt.title('Strip Plot: Type of Travel vs Flight Distance')
plt.xlabel('Type of Travel')
plt.ylabel('Flight Distance')
plt.legend(title='Satisfaction')
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='Type of Travel', y='Departure Delay in Minutes', hue='satisfaction', palette=custom_palette, jitter=True)
plt.title('Strip Plot: Type of Travel vs Departure Delay in Minutes')
plt.xlabel('Type of Travel')
plt.ylabel('Departure Delay (Minutes)')
plt.legend(title='Satisfaction')
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='Type of Travel', y='Arrival Delay in Minutes', hue='satisfaction', palette=custom_palette, jitter=True)
plt.title('Strip Plot: Type of Travel vs Arrival Delay in Minutes')
plt.xlabel('Type of Travel')
plt.ylabel('Arrival Delay (Minutes)')
plt.legend(title='Satisfaction')
plt.show()

#%%

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.stripplot(data=df, x='Type of Travel', y='Flight Distance', hue='satisfaction', palette=custom_palette, jitter=True, ax=axes[0])
axes[0].set_title('Type of Travel vs Flight Distance')
axes[0].set_xlabel('Type of Travel')
axes[0].set_ylabel('Flight Distance')
axes[0].legend(title='Satisfaction')

sns.stripplot(data=df, x='Type of Travel', y='Departure Delay in Minutes', hue='satisfaction', palette=custom_palette, jitter=True, ax=axes[1])
axes[1].set_title('Type of Travel vs Departure Delay in Minutes')
axes[1].set_xlabel('Type of Travel')
axes[1].set_ylabel('Departure Delay (Minutes)')
axes[1].legend(title='Satisfaction')

sns.stripplot(data=df, x='Type of Travel', y='Arrival Delay in Minutes', hue='satisfaction', palette=custom_palette, jitter=True, ax=axes[2])
axes[2].set_title('Type of Travel vs Arrival Delay in Minutes')
axes[2].set_xlabel('Type of Travel')
axes[2].set_ylabel('Arrival Delay (Minutes)')
axes[2].legend(title='Satisfaction')

plt.tight_layout() 
plt.show()


#%%
categories = ['Inflight service', 'Cleanliness', 'Departure/Arrival time convenient']
values_satisfied = [df[df['satisfaction'] == 'satisfied'][cat].mean() for cat in categories]
values_neutral_dissatisfied = [df[df['satisfaction'] == 'neutral or dissatisfied'][cat].mean() for cat in categories]

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

values_satisfied += values_satisfied[:1]
values_neutral_dissatisfied += values_neutral_dissatisfied[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values_satisfied, color='lightblue', alpha=0.5)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title('Satisfaction Levels - Satisfied Customers', fontsize=14)
ax.grid(True)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values_neutral_dissatisfied, color='lightpink', alpha=0.5)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title('Satisfaction Levels - Neutral or Dissatisfied Customers', fontsize=14)
ax.grid(True)

plt.show()
#%%
selected_columns = ['Age', 'Flight Distance', 'Inflight wifi service', 'Food and drink', 'Seat comfort']
sns.pairplot(df[selected_columns])
plt.suptitle('Pair Plot of Selected Columns')
plt.show()
# %%
sns.lmplot(x='Age', y='Arrival Delay in Minutes', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'green'})
plt.title('Regression Plot of Age vs Arrival Delay in Minutes')
plt.xlabel('Age')
plt.ylabel('Arrival Delay in Minutes')
plt.show()


# %%
sns.rugplot(df['Arrival Delay in Minutes'], color='purple')
plt.title('Rug Plot of Arrival Delay in Minutes')
plt.xlabel('Arrival Delay in Minutes')
plt.show()

# %%
features=df.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction', 'Age.cat', 'Age'])
scaler=StandardScaler()
scaled_f=scaler.fit_transform(features)
print(np.round(scaled_f, 2))
# %%
cm=pd.DataFrame(scaled_f, columns=features.columns).corr()
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True)
plt.show()
# %%
u, s, V=svd(scaled_f, full_matrices=False)
cond_num=cond(scaled_f)
print('singular values are')
print(np.round(s,decimals=2))
print('conditional number is ')
print(np.round(cond_num,decimals=2))
# %%
pca=PCA(n_components=0.95)
pca.fit(scaled_f)
explained_variance=pca.explained_variance_ratio_
print(np.round(explained_variance, 2))
print('\n')


components=pca.n_components_
features_to_be_removed=scaled_f.shape[1]-components
print('the number of features to be removed are', features_to_be_removed)
# %%
transformed_f=pca.transform(scaled_f)
u_reduced, s_reduced, v_reduced=svd(transformed_f, full_matrices=False)
print('the new singular values are')
print(np.round(s_reduced, 2))
print('\n')
cond_num_reduced=cond(transformed_f)
print('the new conditional number is')
print(np.round(cond_num_reduced, 2))
# %%
cm_reduced=pd.DataFrame(transformed_f).corr()
plt.figure(figsize=(10,8))
sns.heatmap(cm_reduced, annot=True, fmt='.2f')
plt.show()

# %%
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()
# %%
from scipy.stats import shapiro

feature_names = features.columns.tolist()

shapiro_results = {}
for col in range(transformed_f.shape[1]):
    feature_name = feature_names[col]
    shapiro_stat, shapiro_pvalue = shapiro(transformed_f[:, col])
    shapiro_results[feature_name] = {
        'Shapiro Statistic': round(shapiro_stat, 2),
        'p-value': round(shapiro_pvalue, 2),
        'Normal (p > 0.05)': shapiro_pvalue > 0.05 
    }

for feature, result in shapiro_results.items():
    print(f"{feature}: Shapiro Statistic = {result['Shapiro Statistic']}, p-value = {result['p-value']}, Normality Assumption (p > 0.05) = {result['Normal (p > 0.05)']}")


# %%
scaled_f_df = pd.DataFrame(scaled_f) 
sns.clustermap(scaled_f_df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title('Cluster Map of Correlation Matrix')
plt.show()

# %%
