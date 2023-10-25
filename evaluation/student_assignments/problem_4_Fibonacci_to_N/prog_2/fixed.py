#User function Template for python3

class Solution:
    def nFibonacci(self,n):
            #code here
            k = [0,1]
            a = 0
            b = 1
            c = 0
            for i in range(1,n+1):
                c = a+b
                a=b
                b=c
                if c <= n:
                    k.append(c)
                else:
                    break
            return k



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