#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        ans=[]
        if (len(S1))>(len(S2)):
            l=len(S1)
            s=len(S2)
        else:
            l=len(S2)
            s=len(S1)
        for i in range(s):
            ans.append(S1[i])
            ans.append(S2[i])
            if i==(len(S1)-1) or i==(len(S2)-1):
                if (len(S1))>(len(S2)):
                    l=len(S1)
                    rest=S1[i+1:]
                else:
                    
                    l=len(S2)
                    rest=S2[i+1:]
        re=''.join(ans)
        return(re+rest)


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