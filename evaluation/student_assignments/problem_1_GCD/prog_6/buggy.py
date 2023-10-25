#User function Template for python3

class Solution:
    def gcd(self, A, B):
        # code here
        l1=[]
        for i in range(1,max(A,B)):
            if A%i==0 and B%i==0:
                l1.append(i)
        return max(l1)


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