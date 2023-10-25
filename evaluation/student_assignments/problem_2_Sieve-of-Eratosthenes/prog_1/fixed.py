#User function Template for python3

class Solution:
    def is_prime(self, num):
        sqrt_num = int(num**0.5) + 1
        for i in range(2, sqrt_num):
            if num % i == 0:
                return False
        return True
    def sieveOfEratosthenes(self, N):
        ans = []
        for i in range(2, N+1):
            if self.is_prime(i) == True:
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