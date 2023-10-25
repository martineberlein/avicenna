#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        if N == 1:
            return []

        primes = [0, 0] + [1] * (N + 1)
        primeNums = []
        
        for i in range(2, N+1):
            if primes[i] == 1:
                for j in range(2 * i, N+1, i):
                    primes[j] = 0
        
        for i in range(2, N+1):
            if primes[i] == 1:
                primeNums.append(i)
          
        return primeNums


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