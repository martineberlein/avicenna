class Solution:
    def gcd(self, A, B):
        l=[]
        if A>=B:
            for i in range(1,A):
                if A%i==0 and B%i==0:
                    l.append(i)
            if len(l) <= 1:
                return l[0]
            else:
                return l[-1]
        else:
            for i  in range(1,B):
                 if A%i==0 and B%i==0:
                    l.append(i)
            if len(l) <= 1:
                return l[0]
            else:
                return l[-1]
        





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