#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        A=[]
        if N==0:
            A.append(0)
        elif N==1:
            A.append(0)
            A.append(1)
            A.append(1)
        else:
            n1=0
            n2=1
            A.append(n1)
            A.append(n2)
            while True:
                n=n1+n2
                if n <= N:
                    A.append(n)
                    n1=n2
                    n2=n
                else:
                    break
        return A


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