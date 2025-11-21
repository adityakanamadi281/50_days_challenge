# 1. Find a pair with the given sum in an array

nums = [8, 7, 2, 5, 3, 1]
target = 10

def find_all_pairs(nums, target):
    seen = set()
    pairs = []

    for num in nums:
        diff = target - num
        
        if diff in seen:
            pairs.append((diff, num))
        
        seen.add(num)
    
    return pairs

# Example usage
result = find_all_pairs(nums, target)
print("Pairs found:", result)


# 2 Insertion sorting

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]        
        j = i - 1

     
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key    


nums = [12, 11, 13, 5, 6]
insertion_sort(nums)
print("Sorted array:", nums)
