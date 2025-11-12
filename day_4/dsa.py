stack = []
stack.append(10)
stack.append(20)
stack.append(30)
stack.append(40)
stack.append(50)
print(stack.pop())  # 30 removed first
print(stack)



def reverse_string_using_stack(string):
    
    stack = []

    for char in string:
        stack.append(char)


    reversed_string = ""
    while stack:
        reversed_string += stack.pop()

    return reversed_string


input_string = "Aditya"
print("Original String:", input_string)
print("Reversed String:", reverse_string_using_stack(input_string))






# 2

def calculate_stock_span(prices):
    n = len(prices)
    span = [0] * n  
    stack = []  

    for i in range(n):
        while stack and prices[i] >= prices[stack[-1]]:
            stack.pop()

        
        if not stack:
            span[i] = i + 1
        else:
            span[i] = i - stack[-1]

        stack.append(i)

    return span


prices = [100, 80, 60, 70, 60, 75, 85]
spans = calculate_stock_span(prices)
print("Stock Prices:", prices)
print("Stock Spans:  ", spans)
