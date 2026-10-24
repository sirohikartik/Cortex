import torch
class Node():
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    '''Here i will calculate the entropy of a given set'''
    _,counts = torch.unique(y,return_counts=True)

    probs = counts.float()/len(y)

    return -(torch.sum(probs * torch.log2(probs + 1e-9)))

def info_gain(X_feat,y,threshold):
    '''Here i will find the info gain i.e. reduction in entropy obtained if we split at this function'''

    parent_entropy = entropy(y)

    left_mask = X_feat <= threshold
    right_mask = ~left_mask
    if len(y[left_mask]) ==0 or len(y[right_mask]) ==0:
        return 0
    n = len(y)

    n_l = len(y[left_mask])
    n_r = len(y[right_mask])
    e_l = entropy(y[left_mask]) 
    e_r = entropy(y[right_mask])

    child_entropy = (n_l/n) * e_l + (n_r/n)* e_r
    return parent_entropy - child_entropy

def build_tree(X_train,y,depth=0,max_depth=5):
    n_samples,n_features = X_train.shape
    n_labels = len(torch.unique(y))

    if depth >= max_depth or n_labels ==1 or n_samples ==1:
        leaf_value = torch.mode(y).values.item()
        return Node(value=leaf_value)

    best_gain = -1
    split_idx,split_threshold = None,None
    
    for feat_idx in range(n_features):
        X_column = X_train[:,feat_idx]
        thresholds = torch.unique(X_column)

        for threshold in thresholds:
            gain = info_gain(X_column,y,threshold)
            if gain>best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_threshold = threshold

    if split_idx is None: # Safety check
        return Node(value=torch.mode(y).values.item())

    left_mask = X_train[:, split_idx] <= split_threshold
    right_mask = ~left_mask
    
    left_tree = build_tree(X_train[left_mask], y[left_mask], depth + 1, max_depth)
    right_tree = build_tree(X_train[right_mask], y[right_mask], depth + 1, max_depth)

    return Node(feature=split_idx, threshold=split_threshold, left=left_tree, right=right_tree)
def predict(node,x):
    if node.value is not None:
        return node.value
    if x[node.feature]<=node.threshold:
        return predict(node.left,x)
    return predict(node.right,x)

def run_test():
    print("--- Testing Decision Tree (XOR Problem) ---")
    
    # XOR Data: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
    X_train = torch.tensor([
        [0.0, 0.0], [0.1, 0.1], 
        [0.0, 1.0], [0.1, 0.9],
        [1.0, 0.0], [0.9, 0.1],
        [1.0, 1.0], [0.9, 0.9]
    ])
    y_train = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])

    # Build Tree
    tree_root = build_tree(X_train, y_train, max_depth=3)

    # Test Samples
    test_cases = [
        torch.tensor([0.0, 0.0]), # Expected 0
        torch.tensor([0.0, 1.0]), # Expected 1
        torch.tensor([1.0, 0.0]), # Expected 1
        torch.tensor([1.1, 1.1])  # Expected 0
    ]

    for i, test in enumerate(test_cases):
        res = predict(tree_root, test)
        print(f"Test {i+1} {test.tolist()}: Predicted {int(res)}")

run_test()







         
