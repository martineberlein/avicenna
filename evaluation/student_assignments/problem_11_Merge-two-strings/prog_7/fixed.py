#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        a = list(S1)
        b = list(S2)
        ans = []
        p = len(a)
        m = len(b)
        n = min(p,m)
        for i in range(0,n):
            ans.append(a[i])
            ans.append(b[i])
        ans = ("").join(ans)
        if(p>m):
            ans = ans+S1[-(p-n):]

        elif(p<m):
            ans = ans+S2[-(m-n):]
        return(ans)


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