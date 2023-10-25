#User function Template for python3

class Solution:
    def middle(self,A,B,C):
        if A > B: #A 10
            if A > C:
                if B > C:
                    return B
                return C
            return A
        if A > C:
            return A
        if C > B:
            return B
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