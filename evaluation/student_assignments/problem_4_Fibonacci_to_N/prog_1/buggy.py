#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        f1=0;
        f2=1;
        l=[f1,f2]
        for i in range(1,N+1):
          sum=f1+f2
          l.append(sum)
          f1=f2
          f2=sum
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