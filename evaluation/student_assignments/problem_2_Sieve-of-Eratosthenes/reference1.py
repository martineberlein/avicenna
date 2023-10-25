#User function Template for python3

class Solution:
  def sieveOfEratosthenes(self, N):
    # Create the sieve
    is_prime = [True] * (N+1)
    for i in range(2, N):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    
    # Get the list of primes
    result = []
    for i in range(2, N+1):
        if is_prime[i]:
            result.append(i)
    return result


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