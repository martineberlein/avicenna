#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x): 
        for i in range(1,x):
            if i * i == x:
                return i
            elif i * i > x:
                return i-1
        return


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