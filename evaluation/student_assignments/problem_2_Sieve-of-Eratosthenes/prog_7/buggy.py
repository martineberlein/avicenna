#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        #code here 
        primes=[1]*(N+2)
        primes[0]=primes[1] = 0
        ans=[]
        for i in range(2,N):
            if primes[i]==1:
                for j in range(2*i,N,i):
                    primes[j]=0
        for i in range(len(primes)):
            if primes[i] == 1:ans.append(i)
        return ans 


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