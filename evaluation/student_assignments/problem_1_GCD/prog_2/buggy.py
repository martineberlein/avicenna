#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        a=min(A,B)
        for i in range(a+1,1,-1):
            if(A%i==0 and B%i==0):
                return i
        





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