#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        if A < B:
            A, B = B, A
        while(B):
            A, B = B, A % B
        return A


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