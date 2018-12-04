# Interview Notes

A summary of everything one should know for their technical interview. Made with love by Josh.

- - - -

# Approaching the interview 🚨
1. Ask questions. Find out as much as possible, here are some common things you might want to ask:
	1. Is the data sorted?
	2. Are all values unique? What to do about duplicates?
	3. Any data structure I am not allowed to use?

2. Think about the very simplest case or unit of your problem
	1. What data structure could we use to solve this?
	2. Is this a recursive problem or a Dynamic Programming (DP) problem? 

3. Come up with the obvious brute force solution—don’t worry about coding it but at least start communicating one way of solving the problem.

5. Refine your solution—make it faster and use less space. Perhaps try an alternative, more bold approach.

6. Remember, if your solution starts becoming extremely complicated—you’re doing it wrong. It is often a recursion or DP problem in cases like this.

5. Don’t forget edge cases! What special cases would challenge your solution?

- - - -

# Sorting 🚨
## Merge sort
	* **O(n) space**, thus, bad for arrays
	* **O(n log n) time**
	* Larger K constant than quick-sort
	* Great for forward iterators (linked lists), not random access iterations
	* Can be implemented without extra space w/ linked lists

```python
class MergeSort():
	def __init__(self):
		rr = [12, 11, 13, 5, 6, 7]      	 
    	mergeSort(arr)    
		return

	def mergeSort(arr): 
	    if len(arr) >1: 
	        mid = len(arr)//2 #Finding the mid of the array 
	        L = arr[:mid] # Dividing the array elements  
	        R = arr[mid:] # into 2 halves 
	  
	        mergeSort(L) # Sorting the first half 
	        mergeSort(R) # Sorting the second half 
	  
	        i = j = k = 0
	          
	        # Copy data to temp arrays L[] and R[] 
	        while i < len(L) and j < len(R): 
	            if L[i] < R[j]: 
	                arr[k] = L[i] 
	                i+=1
	            else: 
	                arr[k] = R[j] 
	                j+=1
	            k+=1
	          
	        # Checking if any element was left 
	        while i < len(L): 
	            arr[k] = L[i] 
	            i+=1
	            k+=1
	          
	        while j < len(R): 
	            arr[k] = R[j] 
	            j+=1
	            k+=1
```

## Quick sort
* **O(n^2)** at worst
	* Typically acts as O(n log n)
	* In place sort, doesn't require O(n) space (preferred over merge sort)
	* Radix sort is O( K * n ) and may not be faster
	* Preferred for random access data structures, e.g, arrays
	* **Don’t use when data is in reverse, defaults to O(n^n)**

```python
class QuickSort():
	def __init__(self):		
		arr = [10, 7, 8, 9, 1, 5]		
		self.quickSort(arr, 0, len(arr)-1)
		return 

	def partition(arr,low,high):
	    i = ( low-1 )         
	    pivot = arr[high]     
	    for j in range(low, high):	 	               
	        if arr[j] <= pivot:	         	            
	            i = i+1
	            arr[i],arr[j] = arr[j],arr[i]
	 
	    arr[i+1],arr[high] = arr[high],arr[i+1]
	    return ( i+1 )

	def quickSort(arr,low,high):
	    if low < high:		 		      
	        pi = self.partition(arr,low,high)		
	        self.quickSort(arr, low, pi-1)
	        self.quickSort(arr, pi+1, high)
```


- - - -

# Trees 🚨
## Construction
```python
def constructMaximumBinaryTree(self, nums):
	if nums == []: return None
	node, idx = TreeNode(max(nums)), nums.index(max(nums))            
	node.left = self.constructMaximumBinaryTree(nums[:idx])
	node.right = self.constructMaximumBinaryTree(nums[idx + 1:])            
	return node
```

## DFS
```python
# pre order
print(node.val)
self.dfs(node.left)
self.dfs(node.right)

# in order
self.dfs(node.left)
print(node.val)
self.dfs(node.right)

# post order
self.dfs(node.left)
self.dfs(node.right)
print(node.val)
```

## Level order traversal
```python
def doSomething(root):
	if not root: return None
	queue = [root]
	while queue:
		node = queue.pop()
		# do something here
		queue += filter(None, (node.right, node.left))
```

## Swapping two out of place nodes
```python
class Solution:
    def __init__(self):
        self.first = None
        self.second = None
        self.prev = TreeNode(float('-inf'))
        
    def recoverTree(self, root):        
        if not root: return None  
        self.traverse(root)
        self.first.val, self.second.val = self.second.val, self.first.val
        
    def traverse(self, root):
        if not root: return None
        self.traverse(root.left)
        
        # first node is the one that is bigger than the root
        if not self.first and self.prev.val >= root.val:
            self.first = self.prev
        
        # second element is the last one which prev is > current
        if self.first and self.prev.val >= root.val:
            self.second = root
        
        self.prev = root
        
        self.traverse(root.right)
```

## Lowest common ancestor 
```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):    
        if root in (None, p, q): return root
        L, R = [self.lowestCommonAncestor(i, p, q) for i in (root.left, root.right)]        
        return root if L and R else L or R
```

- - - -

# Graphs 🚨
## Construction
```python
graph = collections.defaultdict(list)
for i, j in list_of_tuple:
	graph[i] += [j]
```

## DFS
```python
visited = [False]*N
def dfs(node):
	if not visited[node]:
		visited[node] = True
		# you have touched an untouched node—do something
		for i in graph[node]:
			dfs(i)
```

## Union-Find
```python
class Union():
    def __init__(self):
        self.union = {}

    def root(self, node):
        if node not in self.union:
		      self.union[node] = node
        while self.union[node] != node:
            node = self.union[node]
        return node
        
class Solution(object):
    def findRedundantConnection(self, edges):
        U = Union()
        for edge in edges:
            if U.root(edge[0]) == U.root(edge[1]):
                return edge
            U.union[U.root(edge[0])] = U.root(edge[1])
```

## Finding prerequisites using a graph
```python
class Solution:
    def findOrder(self, numCourses, prerequisites):
        
        # define variables
        graph = collections.defaultdict(set)
        visited = [0]*numCourses
        result = []
        
        # create graph
        for x, y in prerequisites:
            graph[x].add(y)
            
        # dfs - return false if not possible
        def dfs(node):
            if visited[node] == -1: return False
            if visited[node] == 1: return True
            if not visited[node]:
                visited[node] = -1
                for i in graph[node]:
                    if not dfs(i): return False
                visited[node] = 1
                result.append(node)
            return True
            
        # traverse
        for i in range(numCourses):
            if not dfs(i):
                return []            
        
        return result
```

- - - -

# Linked-lists 🚨
## Removing nth node
```python
class Solution:
    def removeNthFromEnd(self, head, n):
        fast = slow = head # define two pointers

		  # fast pointer gets a head start
        for _ in range(n):
            fast = fast.next

        # Edge case if length is two
        if not fast: return head.next

	      # advance both simultaneously
        while fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next # remove nth pointer 
        return head
```

## Odd then Even linked list
```python
class Solution:
    def oddEvenList(self, head):
        dummy1 = odd = ListNode(0)
        dummy2 = even = ListNode(0)
        
        while head:
            odd.next = head
            even.next = head.next
            
            odd = odd.next
            even = even.next
            
            head = head.next.next if even else None
            
        odd.next = dummy2.next        
        return dummy1.next
```

- - - -

# Dynamic Programming 🚨
## Simple 1D memo (jump game)
This can be solved more effectively with one variable instead of an array. Just for DP purposes.

```python
class Solution:
    def canJump(self, nums):
        dp = [0]*len(nums)
        dp[0] = nums[0]
        
        for i in range(1, len(nums)):
            if dp[i -1] < i: return False
            dp[i] = max(dp[i-1], i + nums[i])
            
        return dp[-1] >= len(nums)-1
```

## Traversing a matrix (Minimum path sum)
We want to update each space in the matrix with the minimum cost needed to get there, then return M[-1][-1]

```python
def minPathSum(self, grid):
    m = len(grid)
    n = len(grid[0])
    for i in range(1, n):
        grid[0][i] += grid[0][i-1]
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]
```

## Finding maximum profit with buy & sell in array
Find the largest peak and smallest valley

```python
class Solution:
    def maxProfit(self, prices, result = 0, price_when_bought=float('inf')):        
        for price in prices:
            price_when_bought = min(price, price_when_bought)
            result = max(result, price-price_when_bought)
        return result
```


## Robbing houses
* Consider options, A: rob house, B: don’t rob house
	* If you rob house, you can’t rob the house before it 
	* `max(house_profit + profit[i - 2], profit[i - 1])`
	* Recurse backwards 
```python
class Solution:
    memo = []
    
    def recurse(self, nums, i):
        if i < 0: return 0
        if Solution.memo[i] >= 0: return Solution.memo[i]
        result = max(self.recurse(nums, i - 2) + nums[i], self.recurse(nums, i - 1))
        Solution.memo[i] = result
        return result
    
    def rob(self, nums):
        Solution.memo = [-1]*len(nums)
        return self.recurse(nums, len(nums)-1)
        
```


## Coin Change
Count how many coins it takes to make a target sum, else return -1.

```python
class Solution:
    def coinChange(self, coins, amount):
        dp = [0] + [float('inf')]*amount        
        for i in range(1, amount+1):
            dp[i] = min([dp[i - c] if i - c >= 0 else float('inf') for c in coins]) + 1            
        return dp[-1] if dp[-1] != float('inf') else -1
```


## Splitting array into two even sum partitions
```python
class Solution:
    def canPartition(self, nums):
        if sum(nums) % 2 != 0: return False        
        target = sum(nums) // 2 
        dp = [True] + [False]*target
        
        for num in nums:
            for i in range(target, 0, -1):
                if i >= num: dp[i] = (dp[i] or dp[i-num])
                    
        return dp[target]
```


