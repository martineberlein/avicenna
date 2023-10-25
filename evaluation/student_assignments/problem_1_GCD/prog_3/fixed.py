class Solution:
    def gcd(self, A, B):
        r = A % B
        while r:
            A, B = B, A % B
            r = A % B
        return B
        





#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        A,B = list(map(int, input().strip().split()))
        ob = Solution()
        print(ob.gcd(A,B))
# } Driver Code Ends