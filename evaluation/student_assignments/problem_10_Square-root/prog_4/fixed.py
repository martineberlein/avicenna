#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x): 
    #Your code here
        l, r = 0.0, float(x + 1)
        while r - l > 0.001:
            mid = (l + r) / 2
            if mid * mid <= x:
                l = mid
            else:
                r = mid
        return int(l+0.001)


#{ 
 # Driver Code Starts
#Initial Template for Python 3

import math



def main():
        T=int(input())
        while(T>0):
            
            x=int(input())
            
            print(Solution().floorSqrt(x))
            
            T-=1


if __name__ == "__main__":
    main()
# } Driver Code Ends