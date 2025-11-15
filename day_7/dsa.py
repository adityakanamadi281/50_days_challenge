#Arrays are one of the most commonly-used data structures
#The elements of an array are stored in contiguous memory locations
#Arrays are of two types : Static and Dynamic
#Static arrays have fixed, pre-defined amount of memory that they can use, whereas in dynamic arrays this is flexible
#In Python we only have dynamic arrays
#Some basic operations and their complexities are given below :

#Look-up/Accses - O(1)
#Push/Pop - O(1)*
#Insert - O(n)
#Delete - O(n)

array = [5,8,2,9,17,43,25,10]

#Look-up/Acces
#Any element of an array can be accessed by its index.
#We just need to ask for the particular index of the element we are interested in and we will get the element in constant time
first_element = array[0]  #This will return the first element of the array, in this case, 5, in O(1) time
sixth_element = array[5]  #sixth-element = 43 Again, in O(1) time


#Push/Pop
#Push corresponds to pushing or adding an element at the end of the array.
#Similarly, pop corresponds to removing the element at the end of the array.
#Since the index of the end of the array is known, finding it and pushing or popping an element will only require O(1) time
array.append(87) #Adds 87 at the end of the array in O(1) time

#In some special cases, the append(push) operation may take greater time. This is because as mentioned earlier, Python has dynamic arrays
#So when an element is to appended and the array is filled, the entire array has to be copied to a new location
#With more space allocated(generally double the space) this time so that more items can be appended.
#Therefore, some individual operations may reuire O(n) time or greater, but when averaged over a large number of operations,
#The complexity can be safely considered to be O(1)

array.pop() #Pops/removes the element at the end of the array in O(1) time.

print(array)


#Insert
#Insert operation inserts an element at the beginning of the array, or at any location specified.
#This is O(n) operation since after inserting the element at the desired location,
#The elements to the right of the array have to be updated with the correct index as they all have shifted by one place.
#This requires looping through the array. Hence, O(n) time.
array.insert(0,50) #Inserts 50 at the beginning of the array and shifts all other elements one place towards right. O(n)
array.insert(4,0) #inserts '0' at index '4', thus shifting all elements starting from index 4 one place towards right. O(n)

print(array)


#Delete
#Similar to insert, it deletes an element from the specified location in O(n) time
#The elements to the right of the deleted element have to shifted one space left, which requires looping over the entire array
#Hence, O(n) time complexity
array.pop(0) #This pops the first element of the array, shifting the remaining elements of the array one place left. O(n)
print(array)
array.remove(17) #This command removes the first occurence of the element 17 in the array, for which it needs to traverse the entire array, which requires O(n) time
print(array)
del array[2:4] #This command deletes elements from position 2 to position 4, again, in O(n) time
print(array)

array.insert(11, 'shushrut')
print(array)
print("-"*100)

l = [5,8,2,9,17,43,25,10]
print(len)
l.insert(9, "shushrut")
len = len(l)
print(len)


a = 15
b = 15
c = 300
d = 300
print(a is b, c is d)





#Find the largest word in a given string
#Examples
#Input: "fun&!! time"
#Output: time

#The simplest and easiest solution that comes to mind is :
#We check for every character if it is an alphanumeric character or not
#If it is, we increase a counter and update a variable which stores the maximum value of counter
#If we encunter a non-alphanumeric character, we reset the counter to zero and start again when the next alpha-numeric character arrives

def easy_longest_word(string):
    count = 0
    maximum = 0
    for char in string:
        if char.isalnum():
            count += 1
        else:
            maximum = max(maximum, count)
            count = 0
    maximum = max(maximum, count)
    return maximum

string = 'fun!@#$# times'
print(easy_longest_word(string))

#This prints the length of the longest word, but after writing this funtion I realized we have to print the  word as well 😂
#We can do that using the same logic as above. Just that we have create two new arrays
#One to hold all the words and another to hold the current word.#Then we'll find the word with maximum length and print that

def naive_longest_word(string):
    count = 0
    maximum = 0
    words = []
    word = []
    for char in string:
        if char.isalnum():
            count += 1
            word.append(char)
        else:
            if word not in words and word:
                words.append(''.join(word))
                print(words)
                print(word)
                word = []
            maximum = max(maximum, count)
            count = 0
    maximum = max(maximum, count)
    if word not in words and word:
        words.append(''.join(word))
        print(words)
        print(word)
    for item in words:
        if len(item) == maximum:
            return item

print(naive_longest_word(string))
#As can be seen, this has become a pretty complicated solution.
#We loop over every character and check if it is an alphanumeric character.
#If yes, we increase count by 1 and append the character to the word list.
#If not, we first check if the word which we have accumulated so far is their in the words list or not.
#If not, we convert the list word into a string using the join method and add the string to the words list
#If yes, then we ignore it. This is done so that same words are not added more than once in the words list
#Then we reset word to an empty list in anticipation of the next word and count to 0.
#This way by the end of the loop, words contains al the words in the string except for the last one, which we add manually after the for loop
#Finally, we check the length of which word is equal to the maximum value, which has been keeping track of the length of the longest word
#And we return the longest word, albeit only the first occurence , if there are more than one words with maximum length.

#The complexity is bad on all fronts. There is a join function used inside a for loop.
#Complexity of .join(string) is O(len(string)). So overall time complexity is O(mn)
#Also, two new arrays are created. So space complexity = O(m + n)


#A different method to solve this problem can be using Regex,or Regular Expressions
#First we split the string into groups of alphanumeric characters
#Then we find the maximum length among all the words
#Finally we find the word corresponding to the maximum length

import re

def regex(string):
    string = re.findall('\w+', string)
    maximum = max([len(item) for item in string])
    for item in string:
        if len(item) == maximum:
            return item
sss = "Hello there how are you"
print(regex(sss))