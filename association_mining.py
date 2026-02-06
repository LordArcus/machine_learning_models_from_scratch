# ==================== ASSOCIATION MINING - APRIORI ALGORITHM ====================
from collections import Counter
from itertools import combinations



# Support: P(X and Y) / Total Transactions
def get_support(item, transactions):
    """
    Calculate support for an itemset.
    Args:
        item: list of items
        transactions: dict with transaction_id as key and list of items as value
    Returns:
        support value (float between 0 and 1)
    """
    if not transactions or not item:
        return 0.0
    count = sum([1 for transaction in transactions.values() if set(item).issubset(set(transaction))])
    return count / len(transactions)


# Confidence: P(Y|X) = support(X and Y) / support(X)
def get_confidence(X, Y, transactions):
    """
    Calculate confidence for a rule X -> Y.
    Args:
        X: list of antecedent items
        Y: consequent item (single item or list)
        transactions: dict with transaction_id as key and list of items as value
    Returns:
        confidence value (float between 0 and 1)
    """
    if not X or not Y or not transactions:
        return 0.0
    
    Y = [Y] if not isinstance(Y, list) else Y
    count_XY = sum([1 for transaction in transactions.values() 
                   if set(X + Y).issubset(set(transaction))])
    count_X = sum([1 for transaction in transactions.values() 
                  if set(X).issubset(set(transaction))])
    return count_XY / count_X if count_X != 0 else 0.0


# Lift: confidence(X -> Y) / support(Y)
def get_lift(X, Y, transactions):
    """
    Calculate lift for a rule X -> Y.
    Args:
        X: list of antecedent items
        Y: consequent item
        transactions: dict with transaction_id as key and list of items as value
    Returns:
        lift value (float)
    """
    if not X or not Y or not transactions:
        return 0.0
        
    confidence_XY = get_confidence(X, Y, transactions)
    support_Y = get_support([Y] if not isinstance(Y, list) else Y, transactions)
    return confidence_XY / support_Y if support_Y != 0 else 0.0


def generate_candidate_itemsets(prev_itemsets):
    """
    Generate candidate k-itemsets from (k-1)-itemsets using Apriori principle.
    Args:
        prev_itemsets: list of frequent (k-1)-itemsets
    Returns:
        list of candidate k-itemsets
    """
    candidates = []
    n = len(prev_itemsets)
    
    # Join step: combine itemsets that share first k-2 items
    for i in range(n):
        for j in range(i + 1, n):
            # Sort itemsets for comparison
            set_i = sorted(prev_itemsets[i])
            set_j = sorted(prev_itemsets[j])
            
            # Check if first k-2 items are same (k-1 itemsets have k-2 common prefix)
            if len(set_i) > 1 and set_i[:-1] == set_j[:-1]:
                # Merge by taking union
                candidate = sorted(list(set(set_i) | set(set_j)))
                if candidate not in candidates:
                    candidates.append(candidate)
            elif len(set_i) == 1:
                # For 2-itemsets, just combine each pair
                candidate = sorted(list(set(set_i) | set(set_j)))
                if candidate not in candidates:
                    candidates.append(candidate)
    
    return candidates


def apriori(transactions, min_support=0.5, min_confidence=0.5, min_lift=1.0):
    """
    Generate association rules using Apriori algorithm.
    
    Apriori Principle: If an itemset is infrequent, all its supersets are infrequent.
    This allows pruning of candidate itemsets before checking support.
    
    Args:
        transactions: dict with transaction_id as key and list of items as value
        min_support: minimum support threshold (0-1)
        min_confidence: minimum confidence threshold (0-1)
        min_lift: minimum lift threshold (default: 1.0)
    
    Returns:
        list of dicts containing: {antecedent, consequent, support, confidence, lift}
    """
    
    if not transactions or not (0 <= min_support <= 1) or not (0 <= min_confidence <= 1):
        return []
    
    # Step 1: Find frequent 1-itemsets (Apriori principle: base case)
    my_items = [item for transaction in transactions.values() for item in transaction]
    items_list = Counter(my_items)
    
    frequent_1_itemsets = []
    for item in items_list.keys():
        support_item = get_support([item], transactions)
        if support_item >= min_support:
            frequent_1_itemsets.append([item])
    
    if not frequent_1_itemsets:
        return []
    
    # Step 2: Generate frequent k-itemsets using Apriori principle
    all_frequent_itemsets = [itemset for itemset in frequent_1_itemsets]
    current_frequent_itemsets = frequent_1_itemsets
    
    k = 2
    while current_frequent_itemsets:
        # Generate candidate k-itemsets from frequent (k-1)-itemsets
        candidate_k_itemsets = generate_candidate_itemsets(current_frequent_itemsets)

        
        if not candidate_k_itemsets:
            break
        
        # Prune candidates: keep only those with min_support
        next_frequent_itemsets = []
        for itemset in candidate_k_itemsets:
            support_val = get_support(itemset, transactions)
            if support_val >= min_support:
                next_frequent_itemsets.append(itemset)
                all_frequent_itemsets.append(itemset)
        
        current_frequent_itemsets = next_frequent_itemsets
        k += 1
    
    # Step 3: Generate rules from frequent itemsets (min_support already met)
    rules = []
    
    # Generate rules from 2-itemsets and larger
    for itemset in all_frequent_itemsets:
        if len(itemset) < 2:
            continue
        
        # Generate all possible antecedent-consequent pairs
        for i in range(1, len(itemset)):
            for antecedent_combo in combinations(itemset, i):
                antecedent = sorted(list(antecedent_combo))
                # Consequent is remaining items
                consequent = sorted(list(set(itemset) - set(antecedent)))
                
                if consequent:  # Ensure consequent is not empty
                    confidence_val = get_confidence(antecedent, consequent, transactions)
                    
                    if confidence_val >= min_confidence:
                        support_itemset = get_support(itemset, transactions)
                        lift_val = get_lift(antecedent, consequent, transactions)
                        
                        if lift_val >= min_lift:
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': round(support_itemset, 4),
                                'confidence': round(confidence_val, 4),
                                'lift': round(lift_val, 4)
                            })
    
    # Remove duplicate rules
    unique_rules = []
    seen = set()
    for rule in rules:
        rule_key = (tuple(rule['antecedent']), tuple(rule['consequent']))
        if rule_key not in seen:
            unique_rules.append(rule)
            seen.add(rule_key)
    
    # Sort by lift (descending)
    unique_rules.sort(key=lambda x: x['lift'], reverse=True)
    
    return unique_rules



################################################################################################
######################### Display ##############################################################
################################################################################################



# my_transactions = {
#     'transaction_1': ['Butter' , 'Bread', "Milk"],
#     'transaction_2': ['Bread', 'Milk'],
#     'transaction_3': ['Butter', 'Milk'],
#     'transaction_4': ['Butter', 'Egg', 'Bread'],
#     'transaction_5': ['Butter', 'Egg', 'Bread', 'Milk']
# }

# my_items = [ item for transaction in my_transactions.values() for item in transaction]
# items_list = Counter(my_items)





def display(rules):
    # Display rules in a formatted way
    print("=" * 80)
    print("ASSOCIATION RULES (Generated by Apriori Algorithm)")
    print("=" * 80)
    for idx, rule in enumerate(rules, 1):
        print(f"\nRule {idx}:")
        print(f"  {rule['antecedent']} → {rule['consequent']}")
        print(f"  Support: {rule['support']:.4f} | Confidence: {rule['confidence']:.4f} | Lift: {rule['lift']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"Total rules found: {len(rules)}")
    print(f"{'=' * 80}")

    # Display as DataFrame for better visualization
    import pandas as pd
    if rules:
        df_rules = pd.DataFrame(rules)
        df_rules['antecedent'] = df_rules['antecedent'].apply(lambda x: ', '.join(x))
        df_rules['consequent'] = df_rules['consequent'].apply(lambda x: ', '.join(x))
        df_rules.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
        print("\nRules as Table:")
        print(df_rules.to_string(index=False))



#----------------------- End------------------------------------------------------------------------------------------------------