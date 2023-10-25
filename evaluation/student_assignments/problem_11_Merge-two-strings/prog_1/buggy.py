#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        a=S1
        b=S2
        c=""
        n=len(a)
        m=len(b)
        i=0
        d=n+m 
        while len(c)<d:
            if i<n:
                c+=a[i]
                if i<m:
                    c+=b[i]
            i+=1 
        return c


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        S1,S2 = map(str,input().strip().split())
        ob = Solution()
        print(ob.merge(S1, S2))
# } Driver Code Ends