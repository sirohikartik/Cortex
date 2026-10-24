import torch

def casual_mask(input_matrix):
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            if i<j :
                input_matrix[i][j] = float('-inf')
    return input_matrix

def efficient_mask(input_matrix):
    '''How to make this efficient'''
    # we have to do this using the boolean mask

    seq_len = input_matrix.shape[-1]
    mask = torch.tril(torch.ones(seq_len,seq_len))
    print("Shape of mask:",mask.shape)
    return input_matrix.masked_fill(mask==0,float('-inf'))

def run_tests(mask_fn):
    print(f"Testing function: {mask_fn.__name__}")
    
    # Test Case 1: 3x3 Matrix
    test_1 = torch.ones(3, 3)
    expected_1 = torch.tensor([
        [1., float('-inf'),float('-inf')],
        [1., 1.,float('-inf')],
        [1., 1., 1.]
    ])
    result_1 = mask_fn(test_1.clone())
    assert torch.equal(result_1, expected_1), f"Test 1 Failed!\nGot:\n{result_1}"
    print("Test 1 (3x3) Passed")

    # Test Case 2: 1x1 Matrix (Edge case)
    test_2 = torch.ones(1, 1)
    expected_2 = torch.tensor([[1.]])
    result_2 = mask_fn(test_2.clone())
    assert torch.equal(result_2, expected_2), "Test 2 Failed!"
    print("Test 2 (1x1) Passed")


    print("\nAll tests passed successfully! ")


if __name__ == "__main__":
    run_tests(efficient_mask)
