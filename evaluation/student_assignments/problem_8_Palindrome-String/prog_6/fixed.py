#User function Template for python3
class Solution:
    def Reverse(self,s,e,S):
        while(s<e):
            temp=S[s]
            S[s]=S[e]
            S[e]=temp
            s=s+1
            e=e-1
        
        return S
    
            
    def isPalindrome(self, S):
        # code here
        S=list(S)
        S_copy = list(S)
        start=0
        end=len(S)-1
        rev=ob.Reverse(start,end,S)
        if(rev==S_copy):
            return 1
        return 0


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
	T=int(input())
	for i in range(T):
		S = input()
		ob = Solution()
		answer = ob.isPalindrome(S)
		print(answer)

# } Driver Code Ends