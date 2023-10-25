#User function Template for python3

class Solution:
    def is_prime(self, num):
        for i in range(2, num):
            if num % i == 0:
                return False
        return True
    def sieveOfEratosthenes(self, N):
        ans = []
        for i in range(2, N):
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