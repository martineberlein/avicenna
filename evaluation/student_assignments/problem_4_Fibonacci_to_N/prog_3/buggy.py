#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        res = []
        a, b = 1, 1
        while a <= N:
            res.append(a)
            a, b = b, a + b
        return res


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__=='__main__':
    t=int(input())
    for _ in range(t):
        N=int(input())
        ob=Solution()
        ans=ob.nFibonacci(N)
        for i in ans:
            print(i,end=" ")
        print()
# } Driver Code Ends