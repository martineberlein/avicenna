#User function Template for python3
from math import sqrt
class Solution:
    def count_divisors(self, N):
        # code here
        #find the divisors
        ans=[]
        count=0
        for i in range(1,int(sqrt(N))+1):
            if(N%i==0):
                ans.append(i)
            if(i!=N//i):
                ans.append(N//i)
        for i in ans:
            if(i%3==0):
                count+=1
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