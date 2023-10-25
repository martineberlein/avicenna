#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        a=len(S1)
        b=len(S2)
        if a>b:
            v="" 
            w=a-b 
            r=S1[:b]
            t=S2 
            d=[(i,j) for i,j in zip(r,t)]
            for i in d:
                for j in i:
                    v+=j 
            return v+S1[-w:] 
        elif a<b:
            v=""
            w=b-a 
            r=S2[:a]
            t=S1
            d=[(i,j)for i,j in zip(t,r)]
            for i in d:
                for j in i:
                    v+=j 
            return v+S2[-w:] 
        else:
            v=""
            r=S1[:b]
            t=S2
            d=[(i,j)for i,j in zip(r,t)]
            for i in d:
                for j in i:
                    v+=j 
            return v 


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