#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x): 
        i=1
        n = (x//2)
        val =1
        while(i<=n):
            if i*i <= x :
                val = i
            else:
                break
            i+=1
    
        return val


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