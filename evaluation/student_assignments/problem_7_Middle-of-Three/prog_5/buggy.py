#User function Template for python3
from math import*
class Solution:
    def middle(self,A,B,C):
        #code here
        if(A>B):
            a=max(A,B)
        else:
            a=min(A,B)
        b=max(B,C)
        return min(a,b)




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