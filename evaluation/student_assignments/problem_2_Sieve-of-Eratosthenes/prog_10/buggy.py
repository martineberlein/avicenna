#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        prime = [True for i in range(N+1)]
        prime[0] = False
        prime[1] = False
        i = 2
        res = []

        for i in range(2, N+1):
            j = 1
            if(prime[i]):
                while(i*j <= N + 1):
                    prime[i*j] = False
                    j += 1
                res.append(i)
        return res


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