import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

df = pd.read_csv('AmesHousing.csv')

categorical_cols = ['Alley', 'Pool QC', 'Fence', 'Fireplace Qu', 'Garage Type',
                    'Garage Finish', 'Garage Qual', 'Garage Cond', 'Bsmt Qual',
                    'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
                    'Mas Vnr Type']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')

numeric_na_to_zero = ['Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Area', 'Garage Cars',
                      'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
                      'Total Bsmt SF', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch',
                      '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val']

for col in numeric_na_to_zero:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
    lambda x: x.fillna(x.median()))

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'SalePrice' in cat_cols:
    cat_cols.remove('SalePrice')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))

numeric_df = df.select_dtypes(include=[np.number])
X = pd.concat([numeric_df.drop('SalePrice', axis=1), encoded_df], axis=1)
y = df['SalePrice']

print(f'Размер матрицы признаков: {X.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

feature_names = X.columns
coef_abs = np.abs(ridge.coef_)
top10_idx = np.argsort(coef_abs)[-10:][::-1]

print('\nТоп-10 самых важных признаков:')
for i, idx in enumerate(top10_idx, 1):
    print(f'{i}. {feature_names[idx]}: {ridge.coef_[idx]:.2f}')

plt.figure(figsize=(10, 6))
plt.scatter(df['Gr Liv Area'], df['SalePrice'], alpha=0.5, s=20, color='purple')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.title('Price vs Living Area')
plt.grid(True, alpha=0.3)
plt.show()

Q1 = df['Gr Liv Area'].quantile(0.25)
Q3 = df['Gr Liv Area'].quantile(0.75)
IQR = Q3 - Q1
iqr_mask = (df['Gr Liv Area'] >= Q1 - 1.5*IQR) & (df['Gr Liv Area'] <= Q3 + 1.5*IQR)

z_scores = np.abs((df['Gr Liv Area'] - df['Gr Liv Area'].mean()) / df['Gr Liv Area'].std())
z_mask = z_scores < 3

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(df[['Gr Liv Area', 'SalePrice']])
iso_mask = iso_labels == 1

clean_mask = iso_mask & iqr_mask & z_mask
df_clean = df[clean_mask].copy()

print(f'\nДо удаления аномалий: {len(df)} записей')
print(f'После удаления аномалий: {len(df_clean)} записей')

X_clean = pd.concat([df_clean.select_dtypes(include=[np.number]).drop('SalePrice', axis=1),
                     pd.get_dummies(df_clean[cat_cols])], axis=1)

X_clean = X_clean.reindex(columns=X.columns, fill_value=0)
y_clean = df_clean['SalePrice']

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42)

scaler_clean = StandardScaler()
X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
X_test_clean_scaled = scaler_clean.transform(X_test_clean)

ridge_orig = Ridge(alpha=1.0)
ridge_orig.fit(X_train_scaled, y_train)
y_pred_orig = ridge_orig.predict(X_test_scaled)

ridge_clean = Ridge(alpha=1.0)
ridge_clean.fit(X_train_clean_scaled, y_train_clean)
y_pred_clean = ridge_clean.predict(X_test_clean_scaled)

print(f'До удаления аномалий - R2: {r2_score(y_test, y_pred_orig):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_orig)):.2f}')
print(f'После удаления аномалий - R2: {r2_score(y_test_clean, y_pred_clean):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)):.2f}')

segmentation_features = ['Gr Liv Area', 'Total Bsmt SF', 'Garage Area', 'Overall Qual',
                         'Overall Cond', 'Year Built', 'Lot Area']

df_seg = df[segmentation_features].copy()
df_seg = df_seg.fillna(df_seg.median())

scaler_seg = StandardScaler()
scaled_seg = scaler_seg.fit_transform(df_seg)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(scaled_seg)

for i in range(5):
    print(f'Сегмент {i}: {len(df[df["Segment"] == i])} объектов')

colors = ['red', 'blue', 'green', 'orange', 'brown']
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(5):
    subset = df[df['Segment'] == i]
    plt.scatter(subset['Gr Liv Area'], subset['Overall Qual'], alpha=0.5, label=f'Segment {i}', s=15, color=colors[i])
plt.xlabel('Gr Liv Area')
plt.ylabel('Overall Qual')
plt.title('Segmentation by Area and Quality')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(5):
    subset = df[df['Segment'] == i]
    plt.scatter(subset['Year Built'], subset['Total Bsmt SF'], alpha=0.5, label=f'Segment {i}', s=15, color=colors[i])
plt.xlabel('Year Built')
plt.ylabel('Total Bsmt SF')
plt.title('Segmentation by Year and Basement Area')
plt.legend()
plt.tight_layout()
plt.show()

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

ridge_pca = Ridge(alpha=1.0)
ridge_pca.fit(X_train_pca, y_train)
y_pred_pca = ridge_pca.predict(X_test_pca)

print(f'R2: {r2_score(y_test, y_pred_pca):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_pca)):.2f}')
print(f'Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.2%}')

df['Age_at_sale'] = df['Yr Sold'] - df['Year Built']
df['Years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
yearly_avg = df.groupby('Yr Sold')['SalePrice'].mean()
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', color='darkgreen', markerfacecolor='red', markersize=8)
plt.xlabel('Year')
plt.ylabel('Avg Price')
plt.title('Price Trend by Year')
plt.xticks(yearly_avg.index)
plt.grid(True, alpha=0.3)

price_2007 = df[df['Yr Sold'] == 2007]['SalePrice'].mean()
price_2008 = df[df['Yr Sold'] == 2008]['SalePrice'].mean()
price_2009 = df[df['Yr Sold'] == 2009]['SalePrice'].mean()
print(f'\nСредние цены: 2007=${price_2007:,.0f}, 2008=${price_2008:,.0f}, 2009=${price_2009:,.0f}')
print(f'Падение 2007-2008: {(price_2008 - price_2007) / price_2007 * 100:.1f}%')

plt.subplot(1, 3, 2)
monthly_avg = df.groupby('Mo Sold')['SalePrice'].mean()
colors_month = plt.cm.RdYlBu(np.linspace(0, 1, 12))
plt.bar(monthly_avg.index, monthly_avg.values, color=colors_month)
plt.xlabel('Month')
plt.ylabel('Avg Price')
plt.title('Seasonality by Month')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)

spring_prices = df[df['Mo Sold'].isin([3, 4, 5])]['SalePrice'].mean()
winter_prices = df[df['Mo Sold'].isin([12, 1, 2])]['SalePrice'].mean()
print(f'Весна: ${spring_prices:,.0f}, Зима: ${winter_prices:,.0f}')
print(f'Разница: {(spring_prices - winter_prices) / winter_prices * 100:.1f}%')

plt.subplot(1, 3, 3)
plt.scatter(df['Age_at_sale'], df['SalePrice'], alpha=0.3, s=10, color='teal')
plt.xlabel('Age at sale (years)')
plt.ylabel('Sale Price')
plt.title('Price vs Age')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f'\nКорреляция возраста с ценой: {df["Age_at_sale"].corr(df["SalePrice"]):.3f}')
print(f'Корреляция лет с ремонта с ценой: {df["Years_since_remod"].corr(df["SalePrice"]):.3f}')

recent_remod = df[df['Years_since_remod'] <= 5]['SalePrice'].mean()
old_remod = df[df['Years_since_remod'] > 20]['SalePrice'].mean()
print(f'Цена с ремонтом до 5 лет: ${recent_remod:,.0f}')
print(f'Цена без ремонта >20 лет: ${old_remod:,.0f}')
