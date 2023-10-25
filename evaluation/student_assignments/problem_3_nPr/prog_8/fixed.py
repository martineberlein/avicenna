#User function Template for python3

class Solution:
    def nPr(self, n, r):
        # code here
        nfact = 1
        for i in range(2,n+1):
            nfact = nfact*i
        
        nrfact = 1
        for i in range(1,n-r+1):
            nrfact *= i
        
        return int(nfact/nrfact)


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