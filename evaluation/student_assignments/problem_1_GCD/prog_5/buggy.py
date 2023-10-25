#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        if(A>B):
            smaller=B
        else:
            smaller=A
        for i in range(1,smaller+1):
            if(A%i==0 and B%i==0):
                gcd=i
        return i



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