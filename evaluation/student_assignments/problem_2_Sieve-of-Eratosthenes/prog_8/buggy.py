#User function Template for python3
import numpy as np
class Solution:
    def sieveOfEratosthenes(self, N):
        #code here 
        sqrt = int(np.sqrt(N))
        is_prime = [True]*N
        
        for i in range(2,sqrt+1):
            if is_prime[i]:
                start = i**2
                while start<N:
                    is_prime[start] = False
                    start += i
        ans = []
        for i in range(2,N):
            if is_prime[i]:
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