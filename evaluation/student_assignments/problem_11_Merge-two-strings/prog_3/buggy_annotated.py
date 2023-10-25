#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # ANNOTATE: initialize the variable `str` to hold the result. You made a bug here: you should actually initialize `str` as an empty string
        str =' '
        # i =0
        # ANNOTATE: loop to add characters from `S1` and `S2` to `str` alternatively
        for i in range(max(len(S1), len(S2))):
            # ANNOTATE: only add `S1[i]` if length of S1 is more than i
            if(i<len(S1)):
                str += S1[i]
            # ANNOTATE: only add `S2[i]` if length of S2 is more than i
            if(i<len(S2)):
                str+= S2[i]
            # ANNOTATE: increment i by 1. This statement is actually redundant, but it doesn't affect the correctness of the program neither
            i+=1    
           
        return str


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