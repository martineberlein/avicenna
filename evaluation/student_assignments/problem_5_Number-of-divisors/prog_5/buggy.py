#User function Template for python3

class Solution:
    def count_divisors(self, N):
        ans = 0
        i = 1
        while i * i <= N:
            if N % i == 0:
                if i % 3 == 0:
                    ans += 1
                elif i != N // i and N // i % 3 == 0:
                    ans += 1
            i += 1
        return ans


#{ 
 # Driver Code Starts
#Initial Template for Python 3#Back-end complete function Template for Python 3#Initial Template for Python 3

if __name__ == '__main__': 
    t = int (input ())
    for _ in range (t):
        N = int(input())
       

        ob = Solution()
        print(ob.count_divisors(N))
# } Driver Code Ends