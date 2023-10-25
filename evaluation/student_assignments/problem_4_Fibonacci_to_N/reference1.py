#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        #code here
        result = [0, 1]
        while True:
            if result[-2] + result[-1] <= N:
                result.append(result[-2] + result[-1])
            else:
                break
        return result



#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__=='__main__':
    t=int(input())
    for _ in range(t):
        N=int(input())
        ob=Solution()
        ans=ob.nFibonacci(N)
        for i in ans:
            print(i,end=" ")
        print()
# } Driver Code Ends