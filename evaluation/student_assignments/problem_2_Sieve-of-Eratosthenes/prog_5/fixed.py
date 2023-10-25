#User function Template for python3

class Solution:
  def sieveOfEratosthenes(self, N):

    prime = [True for i in range(N+1)]
    p = 2
    while (p * p <= N):

        # If prime[p] is not
        # changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * p, N+1, p):
                prime[i] = False
        p += 1

    # Print all prime numbers
    result = []
    for p in range(2, N+1):
        if prime[p]:
            result.append(p)
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