#User function Template for python3
import math
class Solution:
    def count_divisors(self, N):
        # code here
        ans=0
        n=int(math.sqrt(N))
        for i in range(1,n+1):
            if N%i==0:
                if i%3==0:
                    ans+=1
                else:
                    t=N//i
                    if t%3==0:
                        ans+=1
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