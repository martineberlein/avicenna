#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        
        L  = [0, 1]
        if (N==1): 
            return [0,1,1]
        if (N==2): 
            return [0,1,1]
        for i in range(0, N):
            if (L[i]+L[i+1] <= N):
                L.append(L[i]+L[i+1])
            else:
                return(L)
        return(L)


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