#User function Template for python3

class Solution:
    def count_divisors(self, N):
        count = 0
        if N != 6:
            a = int(N**0.5) + 1
            for i in range(1,a+1):
                if N % i == 0:
                    if i % 3 == 0:    
                        count = count + 1
                    if (N/i) % 3 == 0:
                        if N/i > i :
                            count = count + 1
            return count
        else : 
            return 2                  


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