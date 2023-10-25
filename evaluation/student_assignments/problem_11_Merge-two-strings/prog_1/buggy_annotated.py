#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # code here
        # ANNOTATE: initialize `a` and `b` with `S1` and `S2`, respectively
        a=S1
        b=S2
        # ANNOTATE: initialize `c`. `c` will hold the result.
        c=""
        # ANNOTATE: store the length of `a` and `b` in variables `n` and `m`, respectively
        n=len(a)
        m=len(b)
        # ANNOTATE: Initialize `i` as 0. `i` serves as an index, will be used for the loop below.
        i=0
        # ANNOTATE: `d` is total length of the inputs, and is the desired length of the result as well
        d=n+m 
        # ANNOTATE: loop until `c` has length `d`
        while len(c)<d:
            # ANNOTATE: if `n` -- the length of `a`-- is more than `i`, add `a[i]` to the result `c`.
            if i<n:
                c+=a[i]
                # ANNOTATE: if `m` -- the length of `b`-- is more than `i`, add `b[i]` to the result `c`. You made a bug here: this `if` statement should not be inside the body of the `if` statement above. Since you did this, the loop will run forever if the length of S1 is smaller than the length of S2.
                if i<m:
                    c+=b[i]
            # ANNOTATE: increase the index `i` by 1
            i+=1 
        # ANNOTATE: return the result `c`
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