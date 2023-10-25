#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        fabi = []
        
        a = 0
        fabi.append(a)
        b = 1
        fabi.append(b)
        c = 0
        while c<N:
            c = a+b
            fabi.append(c)
            a = b
            b = c
        return fabi


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