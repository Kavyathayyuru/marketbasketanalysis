import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import streamlit as st

# Step 1: Load and Clean Data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'CustomerID'], inplace=True)
    df['Description'] = df['Description'].str.lower()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# Step 2: Exploratory Data Analysis (EDA)
def visualize_top_products(df):
    top_products = df['Description'].value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(y=top_products.index, x=top_products.values, palette='viridis')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Products')
    plt.title('Top 10 Selling Products')
    plt.show()

# Step 3: Market Basket Analysis
def prepare_basket(df):
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

def generate_rules(basket, min_support=0.01, min_confidence=0.2, min_lift=1):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    return rules.sort_values(by='lift', ascending=False)

# Step 4: Product Recommendation
def recommend_products(product_name, rules_df):
    recommendations = rules_df[rules_df['antecedents'].apply(lambda x: product_name in str(x))]
    return recommendations[['consequents', 'confidence', 'lift']]

# Step 5: Visualization
def plot_association_graph(rules):
    G = nx.Graph()
    for i, row in rules.iterrows():
        G.add_edge(str(row['antecedents']), str(row['consequents']), weight=row['lift'])
    
    plt.figure(figsize=(12,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", edge_color="gray")
    plt.title("Market Basket Association Rules")
    plt.show()

# Step 6: Interactive Streamlit Dashboard
def run_streamlit_app():
    st.title("E-commerce Product Recommendation System")
    file_path = st.file_uploader("Upload E-commerce Transactions Data", type=["xlsx", "csv"])
    
    if file_path:
        df = load_data(file_path)
        basket = prepare_basket(df)
        rules = generate_rules(basket)
        
        product = st.text_input("Enter a product name:")
        if st.button("Get Recommendations"):
            recommendations = recommend_products(product, rules)
            st.write(recommendations)

if __name__ == "__main__":
    run_streamlit_app()
