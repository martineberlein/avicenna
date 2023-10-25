#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        # ANNOTATE: initialize `S`, which will hold the final result
        S=""
        # ANNOTATE: save the length of `S1` and `S2` to `a` and `b`, respectively. 
        a = len(S1)
        b = len(S2)
        # ANNOTATE: handle the corner case where 1 of the 2 inputs is empty. Just return the other input.
        if a==0:
            return S2
        elif b==0:
            return S1
            
        # ANNOTATE: add to `S` the characters of the 2 inputs, alternatively, until we reach the end of the shorter one.
        for i in range(min(a,b)):
            S+=S1[i]+S2[i]
        # ANNOTATE: if `S1` is longer than `S2`, add the remaining characters of `S1` to the result
        if a>b:
            S+=S1[b:]
        # ANNOTATE: else, meaning `S2` is longer than or equal to `S1`, add the remaining characters of `S2` to the result. You made a bug here: you mistyped it as `S1` while the correct variable should be `S2`
        else:
            S+=S1[a:]
        # ANNOTATE: return the result `S`
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