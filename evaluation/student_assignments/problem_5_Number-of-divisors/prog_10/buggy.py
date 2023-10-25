#User function Template for python3

class Solution:
    def count_divisors(self, N):
        # code here
        num = []
        i = 1
        while(i<=N):
            if N%i == 0:
                num.append(i)
            i +=1
        ct = 0
        for j in num:
            if j % 3:
                ct +=1
        return ct


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