my_list = [2,4,5,6,8,9,7,3,1]
n=3

for i in range(len(my_list)):
    if my_list[i]==n:
      print(n,"Found at position",i)




def find_min_max(my_list):
    minimum = maximum = my_list[0]
    for num in my_list:
        if num < minimum:
            minimum = num
        elif num > maximum:
            maximum = num
    return minimum, maximum


#arr = [10, 3, 6, 2, 9]
print(find_min_max(my_list))




# count the numbers of elements in a list
nums = [1, 2, 2, 3, 3, 3]
count = {}

for num in nums:
    if num in count:
        count[num] += 1
    else:
        count[num] = 1

print(count)



nums = [1, 2, 3, 4, 2]
seen = set()

for num in nums:
    if num in seen:
        print("Duplicate found:", num)
        break
    seen.add(num)




fruits=["banana","orange","Mango","grapes","kiwi"]

for index , fruit in enumerate(fruits):
    if fruit=="Mango":
        print(index,fruit)
        break


def two_sum(nums, target):
    seen = {}  
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []  


print(two_sum([2, 7, 11, 15], 9))



 

def max_subarray_sum(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  
        max_sum = max(max_sum, window_sum)

    return max_sum


print(max_subarray_sum([2, 1, 5, 1, 3, 2], 3))





def prefix_sum_array(nums):
    prefix = [0] * len(nums)
    prefix[0] = nums[0]

    for i in range(1, len(nums)):
        prefix[i] = prefix[i - 1] + nums[i]
    return prefix

def range_sum(prefix, L, R):
    if L == 0:
        return prefix[R]
    return prefix[R] - prefix[L - 1]

# Example
nums = [2, 4, 6, 8, 10]
prefix = prefix_sum_array(nums)

print(range_sum(prefix, 1, 3))