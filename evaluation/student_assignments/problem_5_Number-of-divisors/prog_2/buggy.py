#User function Template for python3

class Solution:
    def count_divisors(self, n):
        # code here
        c=0
        l=[]
        for i in range(1,n+1):
            if n%i==0 and i%3==0:
                c+=1
        return c


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