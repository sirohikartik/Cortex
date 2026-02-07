import torch
from torch.distributions import Normal

def get_stats(X_train, y_train):
    classes = torch.unique(y_train)
    # Dictionary to store stats: {class_id: (mean_vector, std_vector)}
    stats = {}
    
    for c in classes:
        # Filter X for only this class
        X_c = X_train[y_train == c]
        
        # Calculate mean and std for each feature (column)
        # Using dim=0 to collapse rows
        mean = torch.mean(X_c, dim=0)
        std = torch.std(X_c, dim=0) + 1e-6
        
        stats[c.item()] = (mean, std)
    return stats





def NaiveBayes(X_train,y_train,X_query):
    '''Implementing the Naive Bayes Classifier from scratch'''

    
    # step 1 is to calculate the Prior of the dataset

    classes = torch.unique(y_train)
    priors = {}
    for i in classes:
        count = 0
        for j in y_train:
            if i ==j:
                count+=1
        priors[i.item()] = count/len(y_train)

    # step 2 is to calculate posterior probability (I will use gaussian here)

    ## Basic algo - take the u and \sigma of the data for each class and apply the gaussian pdf on it
    stats = get_stats(X_train,y_train)

    distributions = {}
    for k,v in stats.items():
        distributions[k] = Normal(v[0],v[1])


    final_scores = []

    for idx in range(len(classes)):
        dist = distributions[idx]
        
        log_likelihood = dist.log_prob(X_query)

        score = torch.log(torch.tensor(priors[idx])) + torch.sum(log_likelihood)

        final_scores.append(score)

    return torch.argmax(torch.tensor(final_scores))



#### TESTS #####


def run_tests():
    print("--- Running Naive Bayes Tests ---")
    
    # Generate Synthetic Data
    # Class 0: Center (1,1)
    X0 = torch.randn(50, 2) + torch.tensor([1.0, 1.0])
    y0 = torch.zeros(50)
    
    # Class 1: Center (5,5)
    X1 = torch.randn(50, 2) + torch.tensor([5.0, 5.0])
    y1 = torch.ones(50)
    
    X_train = torch.cat([X0, X1], dim=0)
    y_train = torch.cat([y0, y1], dim=0)

    # Test Case 1: Point clearly in Class 0
    q1 = torch.tensor([1.2, 0.8])
    pred1 = NaiveBayes(X_train, y_train, q1)
    print(f"Test 1 (Near 1,1): Predicted {int(pred1)} | {'✅' if pred1 == 0 else '❌'}")

    # Test Case 2: Point clearly in Class 1
    q2 = torch.tensor([4.8, 5.2])
    pred2 = NaiveBayes(X_train, y_train, q2)
    print(f"Test 2 (Near 5,5): Predicted {int(pred2)} | {'✅' if pred2 == 1 else '❌'}")

    # Test Case 3: Ambiguous point
    q3 = torch.tensor([3.0, 3.0])
    pred3 = NaiveBayes(X_train, y_train, q3)
    print(f"Test 3 (Midpoint 3,3): Predicted {int(pred3)} (Boundary check)")

run_tests()

    















        


    



    
