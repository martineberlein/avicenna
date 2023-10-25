#User function Template for python3

def fact(x):
    if x <= 1:
        return 1
    
    result = 1
    for i in range(2, x+1):
        result *= i
    return result

class Solution:
    def nPr(self, n, r):
        # code here
        return fact(n) // fact(n-r)




#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n, r = [int(x) for x in input().split()]
        
        ob = Solution()
        print(ob.nPr(n, r))
# } Driver Code Ends