import torch
import heapq
'''Implementing KD Tree for efficient nearest neighbour queries'''
import time

class KDNode():
    def __init__(self,point,label,left=None,right=None,axis = 0):
        self.point = point
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis

def build(X_train , y_train ,depth: int = 0):
    '''Implementing KD-Tree for efficient KNN'''
    if X_train.shape[0] ==0 :
        return None
    dim = X_train.shape[1]
    axis = depth%dim
    sorted_idx = torch.argsort(X_train[:,axis])
    X_sorted = X_train[sorted_idx]
    y_sorted = y_train[sorted_idx]
    mid = X_sorted.shape[0]//2
    return KDNode(
    point = X_sorted[mid],
    label= y_sorted[mid],
    axis = axis,
    left = build(X_sorted[:mid],y_sorted[:mid],depth+1),
    right = build(X_sorted[mid+1:],y_sorted[mid+1:],depth +1)
    )


def search_kdtree(query,root,k : int=3):
    '''let's search in this '''
    neighbours = []
    
    def search(node):
        if node is None:
            return 
        dist = torch.dist(query,node.point)
        if len(neighbours) < k :
            heapq.heappush(neighbours, (-dist,node.label))
        elif dist < - neighbours[0][0]:
            heapq.heapreplace(neighbours, (-dist, node.label))
        axis_dist = query[node.axis] - node.point[node.axis]
        near, far = (node.left, node.right) if axis_dist < 0 else (node.right, node.left)
        
        search(near)
        if len(neighbours) < k or abs(axis_dist) < -neighbours[0][0]:
            search(far)
            
    search(root)
    return sorted([(-d, l) for d, l in neighbours])

        
    



    
def brute_force_knn(X_train, y_train, X_query, k=3):
    dists = torch.norm(X_train - X_query, dim=1)
    values, indices = torch.topk(dists, k, largest=False)
    return sorted([(d.item(), y_train[i].item()) for d, i in zip(values, indices)])

# --- TEST RUNNER ---
def run_comparison_tests(num_samples=100000, dim=100, k=3):
    print(f"Testing with {num_samples} samples in {dim}D space (k={k})")
    
    # 1. Generate Random Data
    X_train = torch.rand(num_samples, dim)
    y_train = torch.arange(num_samples) # Unique labels for verification
    X_query = torch.rand(dim)
    
    # 2. Build Tree
    start_build = time.time()
    root = build(X_train, y_train)
    build_time = time.time() - start_build
    
    # 3. Search KD-Tree
    start_kd = time.time()
    kd_res = search_kdtree(X_query, root, k)
    kd_time = time.time() - start_kd
    
    # 4. Search Brute Force
    start_brute = time.time()
    brute_res = brute_force_knn(X_train, y_train, X_query, k)
    brute_time = time.time() - start_brute
    
    # 5. Compare Results
    # We round distances to avoid tiny float precision mismatches
    kd_labels = sorted([res[1] for res in kd_res])
    brute_labels = sorted([res[1] for res in brute_res])
    
    print(f"Build Time: {build_time:.6f}s")
    print(f"KD-Tree Search Time: {kd_time:.6f}s")
    print(f"Brute Force Search Time: {brute_time:.6f}s")
    
    if kd_labels == brute_labels:
        print("✅ SUCCESS: KD-Tree found the exact same neighbors as Brute Force.")
    else:
        print("❌ FAILURE: Results mismatch.")
        print(f"KD-Tree Labels: {kd_labels}")
        print(f"Brute Labels:  {brute_labels}")

# Run the test
run_comparison_tests(num_samples=5000, dim=2, k=3)

    


