#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        prime = [True]*(N+1)
        
        prime[0], prime[1] = False, False
        
        p = 2
        while p**2 <= N:
            
            if prime[p]:
                for i in range(p**2, N+1, p):
                    prime[i] = False
                    
            p += 1
        
        primes = []   
        for i in range(N+1):
            if prime[i]: primes.append(i)
            
        return primes


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