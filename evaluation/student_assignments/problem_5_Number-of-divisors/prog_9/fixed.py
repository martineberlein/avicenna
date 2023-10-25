#User function Template for python3

class Solution:
    def count_divisors(self, N):
        import math
        count = 0
        if N % 3 == 0:
            count += 1
            for i in range(2, int(math.sqrt(N)) + 1):
                if N % i == 0:
                    if (N // i) == i:
                        if i % 3 == 0:
                            count += 1
                    else:
                        if i % 3 == 0:
                            count += 1
                        if (N // i) % 3 == 0:
                            count += 1
        return count


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