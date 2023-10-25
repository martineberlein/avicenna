#User function Template for python3

class Solution:
    def middle(self,A,B,C):
        if B > C and B < A or B < C and B > A:
            return B
        elif A > C and A < B or A < C and A > B:
            return A
        elif C > B and C < A or C < B and C > A:
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