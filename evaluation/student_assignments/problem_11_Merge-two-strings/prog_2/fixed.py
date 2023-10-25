#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        S=""
        a = len(S1)
        b = len(S2)
        if a==0:
            return S2
        elif b==0:
            return S1
            
            
        for i in range(min(a,b)):
            S+=S1[i]+S2[i]
        if a>b:
            S+=S1[b:]
        else:
            S+=S2[a:]
        return S


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