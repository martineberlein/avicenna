#User function Template for python3

class Solution:
    def middle(self,A,B,C):
        #code here
        min=0
        max=0
        mid=0
        if A>B:
            max=A
            min=B
            if A>C:
                if B>C:
                    min=C
                    mid=B
            else:
                max=C
                mid=A
        if A<B:
            min=A
            max=B
            if B>C:
                if A>C:
                    min=C
                    mid=A
                else:
                    min=A
                    mid=C
            else:
                max=C
                mid=B
        return mid


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