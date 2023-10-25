#User function Template for python3

class Solution:
    def sieveOfEratosthenes(self, N):
        #code here
        seive_array = [True]*(N+1)
        seive_array[0] = False
        seive_array[1] = False
        i = 2
        while(i*i<=N):
            if seive_array[i]==True:
                for j in range(i*i,N+1,i):
                    seive_array[j]=False
            i+=1
        list_primes = []
        for i in range(N):
            if seive_array[i]==True:
                list_primes.append(i)
        
        return list_primes


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