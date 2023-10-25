#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        a=''
        c=S1+S2[::-1]
        i=0
        j=len(c)-1
        tis=min([len(S1),len(S2)])
        for k in range(0,tis):
            if i!=j:
                a+=c[i]
                a+=c[j]
            
            i+=1
            j-=1
        if len(S1)>len(S2):
            a=a+S1[tis:]
        else:
            a=a+S2[tis:]
        return a


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