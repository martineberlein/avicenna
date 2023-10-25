#User function Template for python3

class Solution:
    def gcd(self, a, b):
        # code here
        while(1):
            if(a==b):
                return a
            elif(a>b):
                if((a-b)%b==0):
                    return b
                a=a-b
            else:
                if((b-a)%a==0):
                    return a
                b=b-a


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        A,B = list(map(int, input().strip().split()))
        ob = Solution()
        print(ob.gcd(A,B))
# } Driver Code Ends