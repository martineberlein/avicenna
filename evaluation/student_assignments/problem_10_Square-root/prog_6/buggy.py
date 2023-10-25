#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x): 
        low = 1
        high = x
        for i in range(10):
            mid = (low+high)//2
            if mid**2==x:
                return mid
            elif mid**2>x:
                high = mid
            else:
                low = mid
        return low


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