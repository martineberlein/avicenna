#User function Template for python3

class Solution:
    def nPr(self, n, r):
        p = t= 1
        q = n - r
        for i in range(1,n+1):
            p = p * i
        for i in range(1,q+1):
            t = t * i
        k = p/t
        return int(k)
        # code here


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n, r = [int(x) for x in input().split()]
        
        ob = Solution()
        print(ob.nPr(n, r))
# } Driver Code Ends