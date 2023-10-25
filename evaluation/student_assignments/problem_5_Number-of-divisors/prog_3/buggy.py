#User function Template for python3
from math import sqrt
class Solution:
    def count_divisors(self, N):
        s=int(sqrt(N))
        c=0
        for i in range(1,s+1):
            if N%i==0 and i%3==0:
                c+=1
            if i!=N//i and (N//i)%3==0:
                c+=1
        return c
                
                
        
        # code here



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