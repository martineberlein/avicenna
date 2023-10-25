#User function Template for python3
from math import*
class Solution:
    def middle(self,A,B,C):
        #code here
        a = max(A, B, C)
        b = min(A, B, C)
        if A != a and A != b:
            return A
        if B != a and B != b:
            return B
        if C != a and C != b:
            return C




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