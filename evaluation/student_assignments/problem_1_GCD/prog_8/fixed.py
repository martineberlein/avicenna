#User function Template for python3

class Solution:
    
    def gcd(self,a,b):
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return a
    
        if a > b:
            return self.gcd(a%b ,b)
        return self.gcd(a, b % a)




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