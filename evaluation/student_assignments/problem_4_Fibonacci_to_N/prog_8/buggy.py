#User function Template for python3

class Solution:
    def nFibonacci(self,n):
        #code here
        l=[0,1,1]
        if(n==1) :
            return l
        while(l[-1]<=n) :
            l.append(l[-1]+l[-2])
        return l


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