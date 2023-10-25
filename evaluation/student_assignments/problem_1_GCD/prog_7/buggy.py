#User function Template for python3

class Solution:
    def gcd(self, A, B):
        if B==0:
            return A
        else:
            return self.gcd(A,B%A)


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