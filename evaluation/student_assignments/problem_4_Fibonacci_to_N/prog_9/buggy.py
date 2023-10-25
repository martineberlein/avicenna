#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        ans=[]
        if N==0:
            ans.append(0)
        if N==1:
            ans.append(0)
            ans.append(1)
            ans.append(1)
        a=1
        b=1
        c=a+b
        while c<=N:
            ans.append(c)
            a,b=b,c
            c=a+b
        return ans


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