#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        F = [0,1,1]
        n = 2
        lim = 0
        if N<2:
            return F
        else:
            while(lim<N):
                F.append(F[n] + F[n-1])
                n +=1
                lim = F[n] + F[n-1]
        return F


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