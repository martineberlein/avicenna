#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        result=map(lambda x,y:x+y,S1,S2)
        x=list(result)
        for i in range(len(S2),len(S1)):
            x.append(S1[i])
        if len(S1)==0:
            return S2
        elif len(S2)==0:
            return S1
        else:
            return ''.join(x)


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