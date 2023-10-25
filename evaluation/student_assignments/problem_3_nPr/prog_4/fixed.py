#User function Template for python3

class Solution:
    def nPr(self, n, r):
        # code here
        def fact(n):
            if n==0 or n==1:
                return 1
            else:
                res=1
                for i in range(2,n+1):
                    res=res*i
                return res
        if r==0:
            return 1
        # elif r==n-1:
        #     return n
        else:
            num=fact(n)
            den=fact(n-r)
            return num//den


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