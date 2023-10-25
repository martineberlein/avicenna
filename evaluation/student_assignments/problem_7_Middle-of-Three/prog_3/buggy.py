#User function Template for python3

class Solution:
    def middle(self,A,B,C):
        if A-B>=0 and C-B<=0:
            return B
        elif A-C>=0 and B-C<=0:
            return C
        elif B-A>=0 and C-A<=0:
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