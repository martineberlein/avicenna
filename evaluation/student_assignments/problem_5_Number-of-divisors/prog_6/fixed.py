#User function Template for python3

class Solution:
    def count_divisors(self, N):
        if N%3!=0:
            return 0
        x=int(N**0.5)    
        cnt=1
        for  i in range(3,x+1,3):
            if N%i==0:
                cnt+=1
        for i in range(2, x+1):
            if N%i == 0 and i != N//i and (N//i) % 3 == 0:
                cnt+=1
        return cnt            
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