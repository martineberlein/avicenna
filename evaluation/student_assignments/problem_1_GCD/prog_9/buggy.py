#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        while(A!=0 and B!=0):
            if(A>B):
                A=A%B
            else:
                B=B%A
        return B


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        A,B = list(map(int, input().strip().split()))
        ob = Solution()
        print(ob.gcd(A,B))
# } Driver Code Ends