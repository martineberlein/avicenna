#User function Template for python3
import math
class Solution:
    def nPr(self, n, r):
        x=math.factorial(n)
        y=math.factorial(n-r)
        return x//y
        # code here


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