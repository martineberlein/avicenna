#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x):
        if x == 1: return 1
        left,right=0,x 
        while left<right:
            mid=(left+right)//2 
            if mid**2<=x:
                left=mid+1 
            else:
                right=mid 
        return left-1


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