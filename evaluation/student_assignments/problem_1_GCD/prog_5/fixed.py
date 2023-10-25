#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        if A == 0 or B == 0:
            return max(A, B)
        
        if(A>B):
            bigger = A
            smaller=B
        else:
            bigger = B
            smaller=A
        return self.gcd(bigger % smaller, smaller)



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