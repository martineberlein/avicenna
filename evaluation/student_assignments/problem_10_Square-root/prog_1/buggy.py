class Solution:
    def floorSqrt(self, x):
        if x==1:
            return 1
        for i in range(x):
            if i*i==x:
                return i
            elif (i*i)>x:
                return i-1

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