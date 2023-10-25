#User function Template for python3

class Solution:
    def middle(self,A,B,C):
        #code here
        if B in range(A,C):
            return B
        if C in range(A,B):
            return C
        if A in range(B,C):
            return A




#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__=='__main__':
    t=int(input())
    for _ in range(t):
        A,B,C=map(int,input().strip().split())
        ob=Solution()
        print(ob.middle(A,B,C))
# } Driver Code Ends