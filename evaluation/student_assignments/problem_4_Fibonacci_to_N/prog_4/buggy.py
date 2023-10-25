#User function Template for python3

class Solution:
    def nFibonacci(self,N):
        arr = [0,1]
        for i in range(2,N+1):
            if(arr[i-1]+arr[i-2]<=N):
                arr.append(arr[i-1]+arr[i-2])
            else:
                break
        return(arr)


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