#User function Template for python3

class Solution:
    def count_divisors(self, N):
        result = 0
        sqrt_N = int(N**0.5)
        for x in range(1, sqrt_N + 1):
            if N % x == 0:
                if x % 3 == 0:
                    result += 1
                if x * x != N and (N / x) % 3 == 0:
                    result += 1

        return result


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