#User function Template for python3
from math import*

class Solution:
    def checkPrime(self, n):
        if n == 1:
            return False
        if n == 2 or n == 3:
            return True
        if n%2 == 0 or n%3 == 0:
            return False
        for i in range(5,n+1,6):
            if n%i == 0 or n%(i+2) == 0:
                return False
        return True
        
    def sieveOfEratosthenes(self, N):
        #code here 
        dp = [True for i in range(N+1)]
        ans = []
        dp[0] = False
        dp[1] = False
        
        for i in range(2, N+1):
            if dp[i]:
                #ans.append(i)
                for j in range(2*i, N+1, i):
                    dp[j] = False
        
        for i in range(len(dp)):
            if dp[i]:
                ans.append(i)
        
        return ans


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        N = int(input())
        ob = Solution()
        ans = ob.sieveOfEratosthenes(N)
        for i in ans:
            print(i, end=" ")
        print()
# } Driver Code Ends