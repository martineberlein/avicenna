#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # code here
        
        i=0
        j=len(S)
        while i<j:
            if S[i]==S[j]:
                i+=1
                j-=1
            else:
                return False
        return True      


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