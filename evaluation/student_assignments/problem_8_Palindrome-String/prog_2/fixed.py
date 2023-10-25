#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # if len(S) %2 != 0:
        #     return 0
        for i in range(len(S)):
            if S[i] != S[-i-1]:
                return 0
        return 1
                
        # code here


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