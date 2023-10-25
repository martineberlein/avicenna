#User function Template for python3

class Solution:
    def nPr(self, n, r):
        a=b=1
        for i in range(1,n+1):
            if(i<=n-r):
                b*=i
            a*=i
        return(int(a/b))
        
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