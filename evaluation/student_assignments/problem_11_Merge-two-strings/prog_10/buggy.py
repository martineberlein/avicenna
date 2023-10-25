#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        result = ""
        for i in range(min(len(S1),len(S2))):
            result += S1[i] + S2[i]
        if len(S1)>len(S2):
            result += S1[len(S1)-len(S2)+1:]
        elif len(S1)<len(S2):
            result += S2[len(S2)-len(S1)+1:]
        return result


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