#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        t = [1 for i in range(N+1)]
        p = 2
        g = []
        while p*p <= N:
            
            if t[p] == 1:
                
                for i in range(p*p, N+1, p):
                    t[i] = 0
            p+=1
            
        for i in range(2, N+1):
            if t[i] == 1:
                g.append(str(i))
                g.append(" ")
                
        return "".join(g)
                




#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        N = int(input())
        ob = Solution()
        ans = ob.sieveOfEratosthenes(N)
        for i in ans:
            print(i, end=" ")
        print()
# } Driver Code Ends