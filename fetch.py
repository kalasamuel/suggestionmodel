import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from itertools import combinations
from collections import Counter
import sys
import matplotlib.pyplot as plt
import joblib

import os
print("Saving files to:", os.getcwd())

# --- Configuration ---
# Replace with the path to your Firebase service account key file
SERVICE_ACCOUNT_KEY_PATH = 'freshtally8-firebase-adminsdk-fbsvc-aa135aaecb.json' 

# Replace with the SPECIFIC Supermarket ID you want to upload these transactions to
TARGET_SUPERMARKET_ID = "Eqj8gnE1JThFXv73P7t1fmwXpJZ2"

# --- Initialize Firebase Admin SDK ---
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    sys.exit()

# --- Function to fetch data from Firestore ---
def fetch_transactions_from_firestore(db, supermarket_id):
    print(f"Fetching transactions for supermarket: {supermarket_id} from Firestore...")
    transactions_ref = db.collection('supermarkets').document(supermarket_id).collection('pos_transactions')
    docs = transactions_ref.stream()

    data = []
    for doc in docs:
        data.append(doc.to_dict())
    
    print(f"Fetched {len(data)} transactions from Firestore.")
    if not data:
        print("No data fetched from Firestore. Please check your Firestore collection and supermarket ID.")
        return pd.DataFrame() # Return empty DataFrame if no data
    
    return pd.DataFrame(data)

# --- Main script starts here ---

# Load and validate data from Firestore
try:
    df = fetch_transactions_from_firestore(db, TARGET_SUPERMARKET_ID)
    if df.empty:
        sys.exit("Exiting: No data available from Firestore.")

    print("Columns in dataset (from Firestore - BEFORE renaming):", df.columns.tolist())
    
    # Define mapping for column names
    column_mapping = {
        'quantity': 'Quantity',
        'isReturnItem': 'IsReturnItem',
        'transactionId': 'TransactionID',
        'productName': 'ProductName'
    }
    
    # Rename columns
    df.rename(columns=column_mapping, inplace=True)

    print("Columns in dataset (from Firestore - AFTER renaming):", df.columns.tolist())

    # Ensure required columns exist after renaming
    required_columns = ['Quantity', 'IsReturnItem', 'TransactionID', 'ProductName']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns for the ML model after renaming. Needed: {required_columns}")
        print(f"Available columns after renaming: {df.columns.tolist()}")
        sys.exit("Exiting due to missing required columns.")

    # Inspect data after loading from Firestore
    print(f"\nInspecting 'IsReturnItem' column:")
    print("Unique values:", df['IsReturnItem'].value_counts(dropna=False).head(10))
    print(f"Data type: {df['IsReturnItem'].dtype}")

except Exception as e:
    print(f"Error loading data from Firestore or during initial validation: {e}")
    sys.exit()

# Data cleaning
df = df[df['Quantity'] > 0]  # Remove negative quantities
# Ensure 'IsReturnItem' is boolean for filtering
if df['IsReturnItem'].dtype == 'object': # If it's read as object (string 'TRUE'/'FALSE')
    df['IsReturnItem'] = df['IsReturnItem'].astype(str).str.upper() == 'TRUE'
df_sales = df[df['IsReturnItem'] == False].copy()  # Filter non-returns
df_sales = df_sales.drop_duplicates(subset=['TransactionID', 'ProductName'])  # Remove duplicates

# Check transaction size distribution
transaction_sizes = df_sales.groupby('TransactionID')['ProductName'].count()
print("\nTransaction size distribution:")
print(transaction_sizes.value_counts())

# Prepare transactions (only 2+ items)
transactions_for_rules = df_sales.groupby('TransactionID')['ProductName'].apply(list).tolist()
transactions_for_rules = [frozenset(t) for t in transactions_for_rules if len(t) > 1]
print(f"\nPrepared {len(transactions_for_rules)} transactions with 2+ items.")
if not transactions_for_rules:
    print("No transactions with 2+ items found. Consider adjusting filters or using a different dataset.")
    sys.exit()

# Apriori Algorithm
def apriori_algorithm(transactions, min_support):
    item_counts = Counter()
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    
    num_transactions = len(transactions)
    print("\nTop 5 item counts:")
    for itemset, count in item_counts.most_common(5):
        print(f"Itemset: {list(itemset)}, Count: {count}, Support: {count / num_transactions:.4f}")
    
    L1 = {itemset: count for itemset, count in item_counts.items() if (count / num_transactions) >= min_support}
    frequent_itemsets = {0: {itemset: count / num_transactions for itemset, count in L1.items()}}
    print(f"Frequent 1-itemsets (support >= {min_support}): {len(L1)} found")
    
    k = 2
    Lk_minus_1 = L1
    while Lk_minus_1:
        Ck = generate_candidates(Lk_minus_1, k)
        Ck_counts = Counter()
        for transaction in transactions:
            for candidate in Ck:
                if candidate.issubset(transaction):
                    Ck_counts[candidate] += 1
        
        Lk = {itemset: count for itemset, count in Ck_counts.items() if (count / num_transactions) >= min_support}
        if Lk:
            frequent_itemsets[k-1] = {itemset: count / num_transactions for itemset, count in Lk.items()}
            print(f"Frequent {k}-itemsets: {len(Lk)} found")
        else:
            break
        Lk_minus_1 = Lk
        k += 1
    
    return frequent_itemsets

def generate_candidates(Lk_minus_1, k):
    candidates = set()
    list_Lk_minus_1 = list(Lk_minus_1.keys())
    for i in range(len(list_Lk_minus_1)):
        for j in range(i + 1, len(list_Lk_minus_1)):
            itemset1 = list_Lk_minus_1[i]
            itemset2 = list_Lk_minus_1[j]
            union = itemset1.union(itemset2)
            if len(union) == k:
                candidates.add(union)
    
    pruned_candidates = set()
    for candidate in candidates:
        all_subsets_frequent = all(frozenset(subset) in Lk_minus_1 for subset in combinations(candidate, k - 1))
        if all_subsets_frequent:
            pruned_candidates.add(candidate)
    
    return pruned_candidates

def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for k_minus_1 in sorted(frequent_itemsets.keys()):
        if k_minus_1 >= 1:
            for itemset, itemset_support in frequent_itemsets[k_minus_1].items():
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        # Ensure antecedent is in frequent_itemsets before trying to access its support
                        if (len(antecedent) - 1) in frequent_itemsets and antecedent in frequent_itemsets[len(antecedent) - 1]:
                            antecedent_support = frequent_itemsets[len(antecedent) - 1][antecedent]
                            confidence = itemset_support / antecedent_support if antecedent_support > 0 else 0
                            if confidence >= min_confidence:
                                # Ensure consequent is in frequent_itemsets before trying to access its support
                                # Also handle cases where consequent might be a 0-itemset (not in frequent_itemsets[k-1])
                                consequent_support = frequent_itemsets.get(len(consequent) - 1, {}).get(consequent, 0) if len(consequent) > 0 else 1 # If consequent is empty, its support is 1
                                lift = confidence / consequent_support if consequent_support > 0 else float('inf')
                                rules.append({
                                    'antecedents': antecedent,
                                    'consequents': consequent,
                                    'support': itemset_support,
                                    'confidence': confidence,
                                    'lift': lift
                                })
    return pd.DataFrame(rules)

# Parameter tuning
support_values = [0.005, 0.002, 0.001]
MIN_CONFIDENCE = 0.2

for min_support in support_values:
    print(f"\nRunning Apriori with min_support={min_support}")
    frequent_itemsets = apriori_algorithm(transactions_for_rules, min_support=min_support)
    
    # Visualize frequent 1-itemsets with matplotlib
    if frequent_itemsets.get(0):
        itemset_df = pd.DataFrame([{'itemset': ', '.join(sorted(list(k))), 'support': v} for k, v in frequent_itemsets[0].items()])
        itemset_df = itemset_df.sort_values('support', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        plt.bar(itemset_df['itemset'], itemset_df['support'], color='#4e79a7', edgecolor='#2e4977')
        plt.title(f"Top 10 Frequent 1-Itemsets (min_support={min_support})")
        plt.xlabel("Itemset")
        plt.ylabel("Support")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show() # Commented out plt.show() as it can block execution in some environments

    rules_df = generate_rules(frequent_itemsets, min_confidence=MIN_CONFIDENCE)
    
    if not rules_df.empty:
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_df = rules_df.sort_values(by='lift', ascending=False)
        print("\nTop 10 Association Rules:")
        print(rules_df.head(10).to_string())
        rules_df.to_csv(f'supermarket_association_rules_support_{min_support}.csv', index=False)
        joblib.dump(rules_df, 'association_rules.pkl') # Overwrite or save with different name
    else:
        print(f"No rules found with min_support={min_support} and min_confidence={MIN_CONFIDENCE}.")

print("\nModel training process completed using data fetched from Firestore.")