#User function Template for python3

class Solution:
    def nPr(self, n, r):
        # code here
        def fact(x):
            if x == 1:
                return 1
            else:
                val = x # 2
                sec_val = x - 1 # 1
                for i in range(sec_val, 0, -1):
                    val *= sec_val
                    sec_val -=1
                return val
        
        finalVal = fact(n)/fact(n-r)
        return int(finalVal)


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n, r = [int(x) for x in input().split()]
        
        ob = Solution()
        print(ob.nPr(n, r))
# } Driver Code Ends