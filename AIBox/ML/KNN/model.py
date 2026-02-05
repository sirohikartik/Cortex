import torch

## Implementing KNN from scratch 


def KNN_Classifier(X,y,k,X_query):
    '''Implememting KNN classifier from scratch'''

    dist = X - X_query
    dist = torch.sqrt(torch.sum(dist**2,dim=1))
    val,idx = torch.topk(dist,k,largest=False)
    neighbours = y[idx]

    prediction = torch.mode(neighbours,dim=0).values
    return prediction


def KNN_Regressor(X,y,k,X_query):
    '''Implememting KNN classifier from scratch'''

    dist = X - X_query
    dist = torch.sqrt(torch.sum(dist**2,dim=1))
    val,idx = torch.topk(dist,k,largest=False)
    neighbours = y[idx]

    prediction = torch.mean(neighbours.float(),dim=0)
    return prediction



def run_test_classifier(tests=10):
    Xs = [torch.rand(100,3,dtype=torch.float32) for i in range(tests)]
    ys = [ (torch.sum(i,dim=1)>1.5).long() for i in Xs]
    
    for i in range(tests):
        print(f"Test {i+1}",end="")
        try:
            KNN_Classifier(Xs[i],ys[i],3,torch.rand(1,3))
            print(f" :Passed")
        except:
            print(f" :Failed")
    print("Tests Successful")
    
def run_test_regressor(tests=10):
     Xs = [torch.rand(100,3,dtype=torch.float32) for i in range(tests)]
     ys = [torch.mean(i,dim=1).unsqueeze(1) for i in Xs]
     for i in range(tests):
        print(f"Test {i+1}",end="")
        try:
            KNN_Regressor(Xs[i],ys[i],3,torch.rand(1,3))
            print(f" :Passed")
        except Exception as e:
            print(f" :Failed {e} ")
     print("Tests Successful")
    
    



run_test_classifier()
run_test_regressor()
    
    
















